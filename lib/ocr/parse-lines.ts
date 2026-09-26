import type { Receipt, ReceiptItem } from "../../types/domain";
import type { OcrLine, OcrOutput } from "./types";

// Deterministic receipt parser: OCR lines -> Receipt (integer cents) + a
// confidence and a list of warnings for the REVIEW step to highlight. No LLM, no
// network. It is deliberately conservative: a line it isn't sure is an item is
// dropped with a warning (the user can add it), and a total it can't find is
// reported as such — a wrong number on the review screen is worse than a blank.

export interface ParsedItem extends ReceiptItem {
  /** 0..1, from OCR confidence and how cleanly the line parsed. */
  confidence: number;
  /** The OCR text this came from, for the review screen. */
  source: string;
}

export interface ParsedReceipt {
  receipt: Receipt;
  items: ParsedItem[];
  subtotal: number | null;
  totalSource: "keyword" | "computed" | "max-amount" | "none";
  /** True when a currency symbol/code was actually found (else `receipt.currency` is the default). */
  currencyDetected: boolean;
  /** 0..1 overall; below ~0.6 the UI should push Cloud Enhance / manual review. */
  confidence: number;
  warnings: string[];
}

export const DEFAULT_CURRENCY = "RM";

// --- amounts ---------------------------------------------------------------

// 12.50 | 1,234.50 | 1.234,50 | 1 234.50 — exactly two decimals, never glued to more digits.
const AMOUNT = /(?<![\d.,])-?(?:\d{1,3}(?:[.,\s]\d{3})+|\d+)[.,]\d{2}(?!\d)-?/g;

/** "1,234.50" -> 123450, "1.234,50" -> 123450, "12,50" -> 1250. null if not an amount. */
export function parseAmount(token: string): number | null {
  const negative = token.trim().startsWith("-") || token.trim().endsWith("-");
  const clean = token.replace(/[^\d.,\s]/g, "").replace(/\s+/g, "");
  const m = /^(.*)[.,](\d{2})$/.exec(clean);
  if (!m) return null;
  const whole = m[1].replace(/[.,]/g, "");
  if (!/^\d+$/.test(whole)) return null;
  const cents = Number(whole) * 100 + Number(m[2]);
  return negative ? -cents : cents;
}

interface AmountHit {
  cents: number;
  index: number;
  end: number;
  raw: string;
}

function amountsIn(text: string): AmountHit[] {
  const hits: AmountHit[] = [];
  for (const m of text.matchAll(AMOUNT)) {
    const cents = parseAmount(m[0]);
    if (cents !== null) hits.push({ cents, index: m.index, end: m.index + m[0].length, raw: m[0] });
  }
  return hits;
}

/** Every well-formed amount in `text`, in cents (for scoring and sanity checks). */
export function extractAmounts(text: string): number[] {
  return amountsIn(text).map((h) => h.cents);
}

// --- line classification ---------------------------------------------------

/** OCR swaps letters and digits ("T0TAL", "SUBT0TAL", "T0TAL"): fix digits inside a word for keyword matching. */
function forKeywords(text: string): string {
  return text
    .toUpperCase()
    .replace(/(?<=[A-Z])0|0(?=[A-Z])/g, "O")
    .replace(/(?<=[A-Z])1(?=[A-Z])|(?<=[A-Z])\|(?=[A-Z])/g, "I")
    .replace(/[^A-Z0-9%/ ]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

const SUBTOTAL_RE = /\bSUB\s?TOTAL\b|\bSUB\s?TTL\b/;
const TOTAL_RE = /\b(GRAND\s?TOTAL|NET\s?TOTAL|TOTAL\s?(DUE|AMOUNT|PAYABLE|RM|MYR|SALES)?|AMOUNT\s?DUE|BALANCE\s?DUE|JUMLAH|TOTAL)\b/;
const NOT_A_TOTAL_RE = /\bTOTAL\s?(ITEMS?|QTY|QUANTITY|SAVINGS?|DISCOUNT|TAX|SST|GST|VAT|CHANGE|TENDER(ED)?|PAID|POINTS)\b|\bITEMS?\s?TOTAL\b/;
const TAX_RE = /\b(TAX|SST|GST|VAT|SERVICE\s?(CHARGE|CHG|FEE)|SVC(\s?CHG)?|S\/C|SC\s?\d+%?)\b/;
const ROUNDING_RE = /\bROUND(ING|ED)?\b/;
const TENDER_RE = /\b(CASH|CHANGE|TENDER(ED)?|VISA|MASTER(CARD)?|AMEX|CARD|DEBIT|CREDIT|EWALLET|E\s?WALLET|TNG|GRAB\s?PAY|BOOST|QR|PAYMENT|PAID|BALANCE|TIP|GRATUITY)\b/;
const NON_ITEM_RE =
  /\b(TEL|PHONE|FAX|WWW|HTTP|GST\s?(ID|NO|REG)|SST\s?(ID|NO|REG)|INVOICE|RECEIPT|TABLE|PAX|GUEST|CASHIER|SERVER|ORDER|OPEN|THANK|WELCOME|DATE|TIME|STORE|NO\s?\d|ITEMS?\s\d+|POINTS?|MEMBER|CARD\s?NO|REF|APPROVAL|BATCH|TERMINAL)\b/;

type LineKind = "subtotal" | "total" | "tax" | "rounding" | "tender" | "item-candidate";

function classify(text: string): LineKind {
  const k = forKeywords(text);
  if (SUBTOTAL_RE.test(k)) return "subtotal";
  if (TOTAL_RE.test(k) && !NOT_A_TOTAL_RE.test(k)) return "total";
  if (ROUNDING_RE.test(k)) return "rounding";
  if (TAX_RE.test(k)) return "tax";
  if (TENDER_RE.test(k)) return "tender";
  return "item-candidate";
}

const letters = (s: string) => (s.match(/[A-Za-z]/g) ?? []).length;

// --- currency --------------------------------------------------------------

function sniffCurrency(text: string): string | null {
  const t = text.toUpperCase();
  if (/\bRM\s?\d|\bMYR\b|\bRM\b/.test(t)) return "RM";
  if (/S\$|\bSGD\b/.test(t)) return "S$";
  if (/€|\bEUR\b/.test(t)) return "€";
  if (/£|\bGBP\b/.test(t)) return "£";
  if (/\bRP\.?\s?\d|\bIDR\b/.test(t)) return "Rp";
  if (/¥|\bJPY\b/.test(t)) return "¥";
  if (/\$/.test(t) || /\bUSD\b/.test(t)) return "$";
  return null;
}

// --- quantity --------------------------------------------------------------

// "2 @ 0.69", "2 x 3.50", "2X 3.50", "3EA @ 0.29/EA"
const QTY_AT = /(\d{1,3})\s*(?:EA|PCS|X|×|@)\s*(?:@|X|×)?\s*(?:RM|\$)?\s*(\d+[.,]\d{2})/i;
// "2 NASI LEMAK", "2x NASI LEMAK"
const LEADING_QTY = /^\s*(\d{1,2})\s*[xX×]?\s+(?=[A-Za-z])/;

// --- main ------------------------------------------------------------------

interface Row {
  text: string;
  line: OcrLine;
  amounts: AmountHit[];
  /** The line's trailing amount if it sits at the end (allowing a flag letter like "N", "T", "*"). */
  trailing: AmountHit | null;
  kind: LineKind;
  /** Right edge of the line, px, for the price-column check. */
  right: number;
}

function toRow(line: OcrLine): Row {
  const text = line.text.replace(/\s+/g, " ").trim();
  const amounts = amountsIn(text);
  const last = amounts[amounts.length - 1];
  const tail = last ? text.slice(last.end) : "";
  const trailing = last && /^\s*[A-Za-z*]{0,2}\s*$/.test(tail) ? last : null;
  return { text, line, amounts, trailing, kind: classify(text), right: line.box.x + line.box.w };
}

const median = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : 0;
};

export function parseReceiptLines(ocr: Pick<OcrOutput, "lines" | "text" | "width">): ParsedReceipt {
  const warnings: string[] = [];
  const rows = ocr.lines.map(toRow).filter((r) => r.text.length > 0);
  const detectedCurrency = sniffCurrency(ocr.text);

  // Where the item block ends: the first subtotal/total keyword row that has an amount
  // (or is followed by a price-only row, which paddle-style OCR produces).
  const amountFor = (i: number): AmountHit | null => {
    const r = rows[i];
    if (r.trailing) return r.trailing;
    const next = rows[i + 1];
    if (next && letters(next.text) === 0 && next.trailing) return next.trailing;
    return null;
  };

  let subtotalAt = -1;
  let totalAt = -1;
  let grandAt = -1;
  rows.forEach((r, i) => {
    if (r.kind === "subtotal" && subtotalAt < 0 && amountFor(i)) subtotalAt = i;
    if (r.kind === "total" && amountFor(i)) {
      if (totalAt < 0) totalAt = i;
      if (grandAt < 0 && /GRAND\s?TOTAL|AMOUNT\s?DUE/.test(forKeywords(r.text))) grandAt = i;
    }
  });
  const endOfItems = [subtotalAt, totalAt].filter((i) => i >= 0).reduce((a, b) => Math.min(a, b), rows.length);

  // --- items ---------------------------------------------------------------
  const items: ParsedItem[] = [];
  const candidates = rows.slice(0, endOfItems);
  const width = ocr.width || Math.max(1, ...rows.map((r) => r.right));

  // Price column: drop trailing amounts that sit well left of where most prices end
  // (a per-unit "@ 0.49" or a quantity in the middle of a line).
  const rights = candidates.filter((r) => r.trailing && r.kind === "item-candidate").map((r) => r.right);
  const column = median(rights);
  const inColumn = (r: Row) => rights.length < 3 || r.right >= column - 0.25 * width;

  for (let i = 0; i < candidates.length; i++) {
    const r = candidates[i];
    if (r.kind !== "item-candidate" || !r.trailing) continue;
    if (!inColumn(r)) continue;
    const price = r.trailing.cents;
    const before = r.text.slice(0, r.trailing.index).trim();

    // "2 @ 0.69" (or "3EA @ 0.29/EA") under an item: refines THAT item, is not one itself.
    if (letters(before.replace(/\b(EA|PCS|X|@)\b/gi, "")) < 2 && QTY_AT.test(r.text) && items.length > 0) {
      const m = QTY_AT.exec(r.text)!;
      const prev = items[items.length - 1];
      prev.quantity = Number(m[1]);
      prev.unitPrice = parseAmount(m[2]) ?? prev.unitPrice;
      continue;
    }

    let name = before;
    let source = r.text;
    // Name on the line above, price alone on this one (two-line OCR).
    if (letters(name) < 2 && i > 0) {
      const prev = candidates[i - 1];
      if (prev.kind === "item-candidate" && !prev.trailing && letters(prev.text) >= 2) {
        name = prev.text;
        source = `${prev.text} | ${r.text}`;
      }
    }
    if (letters(name) < 2) continue; // a stray number, not an item
    if (NON_ITEM_RE.test(forKeywords(name)) && letters(name) < 12) continue;
    if (price < 0) {
      warnings.push(`Discount/negative line ignored: "${r.text}"`);
      continue;
    }

    let quantity = 1;
    const q = LEADING_QTY.exec(name);
    if (q) {
      quantity = Number(q[1]);
      name = name.slice(q[0].length);
    }
    name = name.replace(/[.\s_~]+$/, "").replace(/\s+/g, " ").trim();
    if (!name) continue;

    items.push({
      name,
      quantity,
      unitPrice: quantity > 1 ? Math.round(price / quantity) : price,
      totalPrice: price,
      confidence: Math.max(0.1, r.line.confidence),
      source,
    });
  }

  // --- totals --------------------------------------------------------------
  const cents = (i: number) => (i >= 0 ? amountFor(i)?.cents ?? null : null);
  const subtotal = cents(subtotalAt);

  let tax = 0;
  let sawTax = false;
  rows.forEach((r, i) => {
    if (r.kind === "tax" && i >= (subtotalAt >= 0 ? subtotalAt : endOfItems - 1)) {
      const c = amountFor(i)?.cents;
      if (c && c > 0) {
        tax += c;
        sawTax = true;
      }
    }
  });

  const itemSum = items.reduce((s, it) => s + it.totalPrice, 0);
  let total: number;
  let totalSource: ParsedReceipt["totalSource"];
  const keywordTotal = cents(grandAt >= 0 ? grandAt : totalAt);
  if (keywordTotal !== null && keywordTotal > 0) {
    total = keywordTotal;
    totalSource = "keyword";
  } else if (items.length > 0) {
    total = (subtotal ?? itemSum) + tax;
    totalSource = "computed";
    warnings.push("No total line found; computed it from the items and tax — please check.");
  } else {
    const all = rows.flatMap((r) => r.amounts.map((a) => a.cents));
    total = all.length ? Math.max(...all) : 0;
    totalSource = all.length ? "max-amount" : "none";
    warnings.push("Could not read any items or a total line.");
  }

  // --- reconciliation & confidence ----------------------------------------
  let confidence = 1;
  const meanOcr = rows.length ? rows.reduce((s, r) => s + r.line.confidence, 0) / rows.length : 0;
  confidence *= 0.5 + 0.5 * meanOcr;
  if (totalSource !== "keyword") confidence -= 0.3;
  if (items.length === 0) confidence -= 0.4;

  const expectedSubtotal = subtotal ?? (keywordTotal !== null ? keywordTotal - tax : null);
  if (expectedSubtotal !== null && items.length > 0) {
    const gap = Math.abs(itemSum - expectedSubtotal);
    if (gap > Math.max(2, expectedSubtotal * 0.01)) {
      confidence -= 0.25;
      warnings.push(
        `Items add up to ${(itemSum / 100).toFixed(2)} but the receipt says ${(expectedSubtotal / 100).toFixed(2)} — a line is probably missing or misread.`,
      );
    }
  }
  if (!sawTax && keywordTotal !== null && subtotal !== null && keywordTotal > subtotal) {
    tax = keywordTotal - subtotal; // tax/service not itemised: it's the gap
  }

  return {
    receipt: {
      items: items.map(({ name, quantity, unitPrice, totalPrice }) => ({ name, quantity, unitPrice, totalPrice })),
      tax,
      total,
      currency: detectedCurrency ?? DEFAULT_CURRENCY,
    },
    items,
    subtotal,
    totalSource,
    currencyDetected: detectedCurrency !== null,
    confidence: Math.max(0, Math.min(1, confidence)),
    warnings,
  };
}
