import { describe, expect, it } from "vitest";
import { parseAmount, parseReceiptLines } from "./parse-lines";
import type { OcrOutput } from "./types";

// Build an OcrOutput from plain text lines: every line sits in the same price
// column (right edge 550 of a 600 px image), OCR confidence 0.9.
function ocr(lines: string[], confidence = 0.9): OcrOutput {
  return {
    text: lines.join("\n"),
    width: 600,
    height: 40 * lines.length + 40,
    ms: 0,
    lines: lines.map((text, i) => ({
      text,
      box: { x: 50, y: 20 + i * 40, w: 500, h: 30 },
      words: [],
      confidence,
    })),
  };
}

describe("parseAmount", () => {
  it.each([
    ["12.50", 1250],
    ["12,50", 1250],
    ["1,234.50", 123450],
    ["1.234,50", 123450],
    ["1 234.50", 123450],
    ["$38.68", 3868],
    ["RM 0.05", 5],
    ["-1.50", -150],
    ["1.50-", -150],
  ])("%s -> %i", (s, cents) => expect(parseAmount(s)).toBe(cents));

  it.each(["abc", "12", "12.5", "12.345", ""])("rejects %j", (s) => expect(parseAmount(s)).toBeNull());
});

describe("parseReceiptLines", () => {
  const mamak = [
    "RESTORAN SRI NASI KANDAR",
    "NO 12 JALAN SS2/24",
    "TEL 03-7877 1234",
    "NASI LEMAK SPECIAL 10.00",
    "2 TEH TARIK 8.00",
    "ROTI CANAI 4.00",
    "SUBTOTAL 22.00",
    "SERVICE CHARGE 10% 2.20",
    "SST 6% 1.45",
    "TOTAL RM 25.65",
    "CASH 30.00",
    "CHANGE 4.35",
  ];

  it("parses a Malaysian receipt end to end, in integer cents", () => {
    const r = parseReceiptLines(ocr(mamak));
    expect(r.receipt.items).toEqual([
      { name: "NASI LEMAK SPECIAL", quantity: 1, unitPrice: 1000, totalPrice: 1000 },
      { name: "TEH TARIK", quantity: 2, unitPrice: 400, totalPrice: 800 },
      { name: "ROTI CANAI", quantity: 1, unitPrice: 400, totalPrice: 400 },
    ]);
    expect(r.subtotal).toBe(2200);
    expect(r.receipt.tax).toBe(365); // service charge + SST
    expect(r.receipt.total).toBe(2565);
    expect(r.receipt.currency).toBe("RM");
    expect(r.currencyDetected).toBe(true);
    expect(r.totalSource).toBe("keyword");
    expect(r.warnings).toEqual([]);
    expect(r.confidence).toBeGreaterThan(0.8);
  });

  it("never treats cash/change/tender lines as items", () => {
    const r = parseReceiptLines(ocr(mamak));
    expect(r.items.map((i) => i.name)).not.toContain("CASH");
    expect(r.items.map((i) => i.name)).not.toContain("CHANGE");
  });

  it("reads OCR'd keywords with digits for letters (T0TAL, SUBT0TAL)", () => {
    const r = parseReceiptLines(ocr(["MEE GORENG 8.00", "SUBT0TAL 8.00", "T0TAL RM 8.00"]));
    expect(r.subtotal).toBe(800);
    expect(r.receipt.total).toBe(800);
    expect(r.totalSource).toBe("keyword");
  });

  it("attaches a '2 @ 0.69' line to the item above instead of making it an item", () => {
    const r = parseReceiptLines(
      ocr(["GROCERY NON TAXABLE 1.38", "2 @ 0.69", "EGGS 3.79", "SUBTOTAL 5.17", "TOTAL $5.17"]),
    );
    expect(r.items).toHaveLength(2);
    expect(r.items[0]).toMatchObject({ name: "GROCERY NON TAXABLE", quantity: 2, unitPrice: 69, totalPrice: 138 });
    expect(r.currencyDetected).toBe(true);
    expect(r.receipt.currency).toBe("$");
    expect(r.warnings).toEqual([]);
  });

  it("joins a name line with a price-only line below it", () => {
    const r = parseReceiptLines(ocr(["CHICKEN RENDANG SET WITH EXTRA", "12.00", "TOTAL 12.00"]));
    expect(r.items).toHaveLength(1);
    expect(r.items[0]).toMatchObject({ name: "CHICKEN RENDANG SET WITH EXTRA", totalPrice: 1200 });
  });

  it("does not mistake 'TOTAL ITEMS 22' or 'TOTAL SAVINGS' for the total", () => {
    const r = parseReceiptLines(
      ocr(["BREAD 3.00", "MILK 4.00", "TOTAL SAVINGS 1.00", "TOTAL ITEMS 2", "SUBTOTAL 7.00", "TOTAL 7.00"]),
    );
    expect(r.receipt.total).toBe(700);
    expect(r.totalSource).toBe("keyword");
  });

  it("warns and lowers confidence when the items don't add up to the subtotal", () => {
    const r = parseReceiptLines(ocr(["A BURGER 10.00", "SUBTOTAL 25.00", "TOTAL 25.00"]));
    expect(r.warnings.some((w) => w.startsWith("Items add up"))).toBe(true);
    const clean = parseReceiptLines(ocr(["A BURGER 10.00", "SUBTOTAL 10.00", "TOTAL 10.00"]));
    expect(r.confidence).toBeLessThan(clean.confidence);
  });

  it("computes a total (and says so) when no total line was read", () => {
    const r = parseReceiptLines(ocr(["A BURGER 10.00", "B FRIES 5.00"]));
    expect(r.receipt.total).toBe(1500);
    expect(r.totalSource).toBe("computed");
    expect(r.warnings.some((w) => w.includes("No total line"))).toBe(true);
    expect(r.confidence).toBeLessThan(0.8);
  });

  it("reports nothing readable instead of inventing items", () => {
    const r = parseReceiptLines(ocr(["asdf", "qwer zxcv"]));
    expect(r.items).toEqual([]);
    expect(r.totalSource).toBe("none");
    expect(r.confidence).toBeLessThan(0.4);
  });

  it("ignores negative (discount) lines with a warning", () => {
    const r = parseReceiptLines(ocr(["BURGER 10.00", "PROMO DISCOUNT -2.00", "SUBTOTAL 10.00", "TOTAL 8.00"]));
    expect(r.items).toHaveLength(1);
    expect(r.warnings.some((w) => w.startsWith("Discount"))).toBe(true);
  });

  it("handles comma decimals and thousand separators", () => {
    const r = parseReceiptLines(ocr(["LAPTOP BAG 1.234,50", "MOUSE 12,50", "TOTAL 1.247,00"]));
    expect(r.items.map((i) => i.totalPrice)).toEqual([123450, 1250]);
    expect(r.receipt.total).toBe(124700);
  });

  it("derives tax as the total-minus-subtotal gap when it isn't itemised", () => {
    const r = parseReceiptLines(ocr(["A 10.00", "SUBTOTAL 10.00", "TOTAL 10.60"]));
    expect(r.receipt.tax).toBe(60);
  });

  it("drops a per-unit amount in the middle of a line (outside the price column)", () => {
    const lines = ocr(["BANANAS 0.20", "COFFEE 4.00", "TEA 3.00", "SUBTOTAL 7.20", "TOTAL 7.20"]);
    // a stray '@ 0.49' fragment that ends far to the left of the price column
    lines.lines.push({ text: "@ 1 lb 0.49", box: { x: 50, y: 400, w: 120, h: 30 }, words: [], confidence: 0.9 });
    lines.lines.sort((a, b) => a.box.y - b.box.y);
    const r = parseReceiptLines(lines);
    expect(r.items.map((i) => i.totalPrice)).toEqual([20, 400, 300]);
  });
});
