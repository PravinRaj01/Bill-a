import { describe, expect, it } from "vitest";
import { centsToInput, formatMoney, fromCents, parseMoneyInput, toCents } from "./money";

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

  it("puts a space after a currency written as a word, but not a symbol or two-letter code", () => {
    expect(formatMoney(9745, "Euro")).toBe("Euro 97.45");
    expect(formatMoney(5000, "USD")).toBe("USD 50.00");
    expect(formatMoney(9745, "RM")).toBe("RM97.45");
    expect(formatMoney(9745, "€")).toBe("€97.45");
  });
});

describe("parseMoneyInput / centsToInput", () => {
  it.each([
    ["12.5", 1250],
    ["12,5", 1250],
    ["  3.07 ", 307],
    ["0", 0],
    ["1.005", 100],
    ["", 0],
    ["abc", 0],
    ["-3", 0],
    ["Infinity", 0],
  ])("parseMoneyInput(%j) = %s", (raw, cents) => {
    expect(parseMoneyInput(raw)).toBe(cents);
  });

  it("round-trips through the input text", () => {
    for (const c of [0, 5, 29, 1999, 123456]) expect(parseMoneyInput(centsToInput(c))).toBe(c);
  });
});
