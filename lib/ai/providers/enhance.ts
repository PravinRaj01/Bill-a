import { z } from "zod";
import type { ParsedItem, ParsedReceipt } from "@/lib/ocr/parse-lines";
import { geminiFailure, geminiUrl, readGeminiText } from "./gemini";
import { MODELS } from "./models";
import { callFetch, ProviderError, type CallContext } from "./types";

// "Cloud Enhance": when the local OCR read is poor, the user can choose to send the
// (already-shrunk) receipt photo to Gemini with THEIR OWN key and get a better
// transcription. It is opt-in per scan, never automatic — the photo leaves the
// device, and on Gemini's free tier Google may use prompts to improve its products.
//
// Gemini transcribes; it does not calculate. Prices are copied as printed, converted
// to integer cents here, and the result is reconciled against the receipt's own
// total so a misread is flagged instead of trusted.

const PROMPT =
  "Transcribe this receipt exactly as printed. Do NOT calculate, correct or infer any number.\n" +
  "- items: every purchased line, with its name, its quantity (1 if not shown) and the line's total price as printed.\n" +
  "- Do not list subtotal, tax, service charge, total, rounding, cash, change or payment lines as items.\n" +
  "- tax: the sum of tax / service-charge lines (SST, GST, VAT, service charge) as printed; 0 if there are none.\n" +
  "- subtotal: the printed subtotal before tax/service, or 0 if none is printed.\n" +
  "- amountTendered: the cash or card amount paid/handed over, or 0 if none is printed.\n" +
  "- total: the grand total of the goods (the amount due, often labelled TOTAL or GRAND TOTAL). It is NOT the cash handed over, the change, or the card amount tendered — those come AFTER the total.\n" +
  "- currency: the currency symbol or code printed (e.g. RM, $), or an empty string.\n" +
  "If the photo is not a receipt, return an empty items list and a total of 0.";

const RESPONSE_SCHEMA = {
  type: "object",
  properties: {
    currency: { type: "string" },
    items: {
      type: "array",
      items: {
        type: "object",
        properties: {
          name: { type: "string" },
          quantity: { type: "integer", minimum: 1 },
          totalPrice: { type: "number" },
        },
        required: ["name", "quantity", "totalPrice"],
      },
    },
    tax: { type: "number" },
    subtotal: { type: "number" },
    total: { type: "number" },
    amountTendered: { type: "number" },
  },
  required: ["currency", "items", "tax", "subtotal", "total", "amountTendered"],
} as const;

const answerSchema = z.object({
  currency: z.string().max(8).default(""),
  subtotal: z.number().finite().min(0).max(10_000_000).default(0),
  amountTendered: z.number().finite().min(0).max(10_000_000).default(0),
  items: z
    .array(
      z.object({
        name: z.string().trim().min(1).max(200),
        quantity: z.number().int().min(1).max(999),
        totalPrice: z.number().finite().min(0).max(10_000_000),
      }),
    )
    .max(300),
  tax: z.number().finite().min(0).max(10_000_000),
  total: z.number().finite().min(0).max(10_000_000),
});

const cents = (n: number) => Math.round(n * 100);

export function toBase64(bytes: Uint8Array): string {
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(bin);
}

export function buildEnhanceBody(imageBase64: string, mimeType = "image/jpeg") {
  return {
    contents: [{ role: "user", parts: [{ text: PROMPT }, { inlineData: { mimeType, data: imageBase64 } }] }],
    generationConfig: {
      temperature: 0,
      maxOutputTokens: 8192,
      responseMimeType: "application/json",
      responseSchema: RESPONSE_SCHEMA,
    },
  };
}

/** Turns Gemini's transcription into the same ParsedReceipt the local parser produces. */
export function receiptFromAnswer(raw: unknown): ParsedReceipt {
  let value = raw;
  if (typeof raw === "string") {
    try {
      value = JSON.parse(raw);
    } catch {
      throw new ProviderError("gemini", "bad-output", "Gemini's transcription wasn't valid JSON.");
    }
  }
  const parsed = answerSchema.safeParse(value);
  if (!parsed.success) throw new ProviderError("gemini", "bad-output", "Gemini's transcription had the wrong shape.");
  const a = parsed.data;

  const items: ParsedItem[] = a.items.map((it) => {
    const total = cents(it.totalPrice);
    return {
      name: it.name,
      quantity: it.quantity,
      unitPrice: it.quantity > 1 ? Math.round(total / it.quantity) : total,
      totalPrice: total,
      confidence: 0.9,
      source: "gemini",
    };
  });

  const warnings: string[] = [];
  const itemSum = items.reduce((s, i) => s + i.totalPrice, 0);
  const taxC = cents(a.tax);
  let totalC = cents(a.total);
  const subtotalC = cents(a.subtotal);
  const tenderedC = cents(a.amountTendered);
  let confidence = 0.9;

  // Gemini sometimes reads the CASH/tendered line as the total (seen live: a Trader Joe's
  // receipt with TOTAL $38.68 and CASH $40.00 came back with total 40.00). Prompting
  // alone didn't stop it, so correct it deterministically — but only when the evidence
  // agrees: it gave the same figure for total and amount tendered, AND the items add up
  // to the printed subtotal. Otherwise leave the total alone and let the check below warn.
  const tol = Math.max(2, subtotalC * 0.01);
  if (tenderedC > 0 && totalC === tenderedC && subtotalC > 0 && Math.abs(itemSum - subtotalC) <= tol && totalC !== subtotalC + taxC) {
    totalC = subtotalC + taxC;
  }

  if (items.length === 0) {
    confidence = 0.1;
    warnings.push("Gemini couldn't find any items either — is this a receipt photo?");
  } else if (totalC > 0 && Math.abs(itemSum + taxC - totalC) > Math.max(2, totalC * 0.01)) {
    confidence = 0.6;
    warnings.push(
      `Items + tax come to ${((itemSum + taxC) / 100).toFixed(2)} but the receipt total was read as ${(totalC / 100).toFixed(2)} — please check.`,
    );
  }

  return {
    receipt: {
      items: items.map(({ name, quantity, unitPrice, totalPrice }) => ({ name, quantity, unitPrice, totalPrice })),
      tax: taxC,
      total: totalC > 0 ? totalC : itemSum + taxC,
      currency: a.currency.trim() || "RM",
    },
    items,
    subtotal: null,
    totalSource: totalC > 0 ? "keyword" : "computed",
    currencyDetected: a.currency.trim().length > 0,
    confidence,
    warnings,
  };
}

export async function enhanceReceipt(
  image: Blob,
  apiKey: string,
  ctx: CallContext = {},
): Promise<ParsedReceipt> {
  const bytes = new Uint8Array(await image.arrayBuffer());
  const res = await callFetch(
    "gemini",
    geminiUrl(ctx.model ?? MODELS.gemini.primary),
    {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-goog-api-key": apiKey },
      body: JSON.stringify(buildEnhanceBody(toBase64(bytes), image.type || "image/jpeg")),
    },
    ctx,
  );
  if (!res.ok) throw await geminiFailure(res);
  return receiptFromAnswer(readGeminiText(await res.json()));
}
