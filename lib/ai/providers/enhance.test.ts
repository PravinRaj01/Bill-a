import { describe, expect, it, vi } from "vitest";
import { buildEnhanceBody, enhanceReceipt, receiptFromAnswer, toBase64 } from "./enhance";
import { geminiUrl } from "./gemini";

const answer = (over: Record<string, unknown> = {}) => ({
  currency: "RM",
  subtotal: 18,
  amountTendered: 0,
  items: [
    { name: "NASI LEMAK", quantity: 1, totalPrice: 10 },
    { name: "TEH TARIK", quantity: 2, totalPrice: 8 },
  ],
  tax: 1.8,
  total: 19.8,
  ...over,
});

describe("receiptFromAnswer", () => {
  it("converts printed amounts to integer cents and reconciles", () => {
    const r = receiptFromAnswer(answer());
    expect(r.receipt.items).toEqual([
      { name: "NASI LEMAK", quantity: 1, unitPrice: 1000, totalPrice: 1000 },
      { name: "TEH TARIK", quantity: 2, unitPrice: 400, totalPrice: 800 },
    ]);
    expect(r.receipt.tax).toBe(180);
    expect(r.receipt.total).toBe(1980);
    expect(r.receipt.currency).toBe("RM");
    expect(r.confidence).toBeGreaterThan(0.8);
    expect(r.warnings).toEqual([]);
  });

  it("no float drift: 19.99 and 0.1+0.2 style values land on exact cents", () => {
    const r = receiptFromAnswer(answer({ items: [{ name: "A", quantity: 1, totalPrice: 19.99 }, { name: "B", quantity: 1, totalPrice: 0.3 }], tax: 0, total: 20.29 }));
    expect(r.receipt.items.map((i) => i.totalPrice)).toEqual([1999, 30]);
    expect(Number.isInteger(r.receipt.total)).toBe(true);
    expect(r.warnings).toEqual([]);
  });

  it("flags a transcription that doesn't add up instead of trusting it", () => {
    const r = receiptFromAnswer(answer({ total: 25 }));
    expect(r.warnings[0]).toMatch(/come to 19\.80 but the receipt total was read as 25\.00/);
    expect(r.confidence).toBeLessThan(0.7);
  });

  it("says so when Gemini finds nothing (not a receipt)", () => {
    const r = receiptFromAnswer(answer({ items: [], tax: 0, total: 0 }));
    expect(r.items).toEqual([]);
    expect(r.confidence).toBeLessThan(0.3);
    expect(r.warnings[0]).toMatch(/couldn't find any items/);
  });

  it("accepts the raw JSON string a provider returns", () => {
    expect(receiptFromAnswer(JSON.stringify(answer())).receipt.total).toBe(1980);
  });

  it.each([
    ["not JSON", "nope"],
    ["missing items", { currency: "RM", tax: 0, total: 1 }],
    ["negative price", answer({ items: [{ name: "A", quantity: 1, totalPrice: -1 }] })],
    ["non-integer quantity", answer({ items: [{ name: "A", quantity: 1.5, totalPrice: 1 }] })],
    ["NaN-ish total", answer({ total: "12.50" })],
    ["absurd price", answer({ items: [{ name: "A", quantity: 1, totalPrice: 1e12 }] })],
  ])("rejects %s", (_n, raw) => {
    expect(() => receiptFromAnswer(raw)).toThrowError(/Gemini/);
  });

  it("defaults the currency when none was printed, and says it wasn't detected", () => {
    const r = receiptFromAnswer(answer({ currency: "" }));
    expect(r.receipt.currency).toBe("RM");
    expect(r.currencyDetected).toBe(false);
  });
});

describe("cash misread as the total (seen live on a Trader Joe's receipt)", () => {
  // TOTAL $38.68, CASH $40.00: Gemini transcribed total = 40.00 = amountTendered
  const tj = (over: Record<string, unknown> = {}) =>
    answer({
      currency: "$",
      items: [{ name: "A", quantity: 1, totalPrice: 20 }, { name: "B", quantity: 1, totalPrice: 18.68 }],
      subtotal: 38.68, tax: 0, total: 40, amountTendered: 40, ...over,
    });

  it("is corrected to subtotal + tax when total == amount tendered AND the items add up to the subtotal", () => {
    const r = receiptFromAnswer(tj());
    expect(r.receipt.total).toBe(3868);
    expect(r.warnings).toEqual([]);
    expect(r.confidence).toBeGreaterThan(0.8);
  });

  it("includes tax in the corrected total", () => {
    const r = receiptFromAnswer(tj({ tax: 2.32, total: 41, amountTendered: 41 }));
    expect(r.receipt.total).toBe(4100); // 38.68 + 2.32 == 41.00: the "wrong" figure was actually right
    expect(r.warnings).toEqual([]);
  });

  it("is NOT corrected when the items don't add up to the subtotal (a line may be missing)", () => {
    const r = receiptFromAnswer(tj({ items: [{ name: "A", quantity: 1, totalPrice: 20 }] }));
    expect(r.receipt.total).toBe(4000);
    expect(r.warnings.length).toBe(1);
  });

  it("is NOT corrected when total and amount tendered differ", () => {
    const r = receiptFromAnswer(tj({ total: 39.5, amountTendered: 40 }));
    expect(r.receipt.total).toBe(3950);
    expect(r.warnings.length).toBe(1);
  });

  it("is NOT corrected when no subtotal was printed", () => {
    const r = receiptFromAnswer(tj({ subtotal: 0 }));
    expect(r.receipt.total).toBe(4000);
  });

  it("older answers without the new fields still parse", () => {
    const { subtotal, amountTendered, ...legacy } = answer();
    void subtotal; void amountTendered;
    expect(receiptFromAnswer(legacy).receipt.total).toBe(1980);
  });
});

describe("enhanceReceipt", () => {
  it("sends the photo + key to Gemini only, asking it to transcribe, not calculate", async () => {
    const f = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ candidates: [{ content: { parts: [{ text: JSON.stringify(answer()) }] }, finishReason: "STOP" }] })),
    );
    const photo = new Blob([new Uint8Array([1, 2, 3, 4])], { type: "image/jpeg" });
    const r = await enhanceReceipt(photo, "AIzaSECRET", { fetchImpl: f as never });
    expect(r.receipt.total).toBe(1980);

    const [url, init] = f.mock.calls[0];
    expect(url).toBe(geminiUrl("gemini-3.5-flash-lite"));
    expect(String(url)).not.toContain("AIzaSECRET");
    const body = JSON.parse(String((init as RequestInit).body));
    const parts = body.contents[0].parts;
    expect(parts[0].text).toMatch(/Do NOT calculate/);
    expect(parts[1].inlineData).toEqual({ mimeType: "image/jpeg", data: toBase64(new Uint8Array([1, 2, 3, 4])) });
    expect(body.generationConfig.responseMimeType).toBe("application/json");
  });

  it("a 401 surfaces as an auth error the UI can explain", async () => {
    const f = vi.fn().mockResolvedValue(new Response("{}", { status: 401 }));
    await expect(enhanceReceipt(new Blob(["x"], { type: "image/jpeg" }), "k", { fetchImpl: f as never })).rejects.toMatchObject({ kind: "auth" });
  });
});

describe("toBase64", () => {
  it("matches Buffer's base64, including for inputs larger than one chunk", () => {
    const small = new Uint8Array([0, 255, 128, 7]);
    expect(toBase64(small)).toBe(Buffer.from(small).toString("base64"));
    const big = new Uint8Array(100_000).map((_, i) => i % 251);
    expect(toBase64(big)).toBe(Buffer.from(big).toString("base64"));
  });

  it("buildEnhanceBody keeps the image out of the text part", () => {
    const b = buildEnhanceBody("QUJD") as any;
    expect(b.contents[0].parts[0].text).not.toContain("QUJD");
  });
});
