import { describe, expect, it } from "vitest";
import {
  formatMoney,
  fromCents,
  receiptToDomain,
  receiptToLegacy,
  splitsToDomain,
  splitsToLegacy,
  toCents,
} from "./money";

describe("toCents / fromCents", () => {
  it.each([
    [0, 0],
    [10, 1000],
    [12.34, 1234],
    [0.29, 29], // 0.29*100 = 28.999999999999996 in floating point
    [1.005, 100], // banker-safe enough: documents that sub-cent input is rounded
    [19.99, 1999],
    [-3, -300], // discount lines
  ])("toCents(%s) = %s", (rm, cents) => {
    expect(toCents(rm)).toBe(cents);
  });

  it("never produces a non-integer, even for floating-point-hostile inputs", () => {
    for (const rm of [0.1 + 0.2, 4.35, 8.2, 1.15, 33.33, 0.07, 100.1]) {
      expect(Number.isInteger(toCents(rm))).toBe(true);
    }
  });

  it("returns 0 for NaN/Infinity instead of poisoning a total", () => {
    expect(toCents(NaN)).toBe(0);
    expect(toCents(Infinity)).toBe(0);
  });

  it("round-trips cents exactly", () => {
    for (const c of [0, 1, 29, 1999, 123456]) expect(toCents(fromCents(c))).toBe(c);
  });

  it("formats", () => {
    expect(formatMoney(1050)).toBe("RM10.50");
    expect(formatMoney(5, "$")).toBe("$0.05");
  });
});

describe("legacy <-> domain conversion", () => {
  const legacy = {
    items: [
      { name: "NASI LEMAK", quantity: 1, unit_price: 10, total_price: 10 },
      { name: "TEH TARIK", quantity: 3, unit_price: 4.33, total_price: 12.99 },
    ],
    tax: 2.3,
    total: 25.29,
    currency: "RM",
  };

  it("converts a receipt to integer cents with camelCase keys", () => {
    const d = receiptToDomain(legacy);
    expect(d.items[1]).toEqual({ name: "TEH TARIK", quantity: 3, unitPrice: 433, totalPrice: 1299 });
    expect(d.tax).toBe(230);
    expect(d.total).toBe(2529);
  });

  it("round-trips a receipt without drift", () => {
    expect(receiptToLegacy(receiptToDomain(legacy))).toEqual(legacy);
  });

  it("converts and round-trips splits", () => {
    const s = [{ name: "Pravin", amount: 18.34, items: "x" }];
    expect(splitsToDomain(s)[0].amount).toBe(1834);
    expect(splitsToLegacy(splitsToDomain(s))).toEqual(s);
  });
});
