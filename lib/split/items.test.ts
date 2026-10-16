import { describe, expect, it } from "vitest";

import { parseItems } from "./items";

describe("parseItems", () => {
  it("reads the engine's item list", () => {
    expect(parseItems("Nasi Lemak (x1/3), Teh (x1), Rice (x2), tax/service")).toEqual([
      { name: "Nasi Lemak", share: "1/3" },
      { name: "Teh" },
      { name: "Rice", share: "×2" },
      { name: "tax/service" },
    ]);
  });
  it("keeps commas inside brackets together", () => {
    expect(parseItems("Combo (a, b) (x1/2), Fries")).toEqual([{ name: "Combo (a, b)", share: "1/2" }, { name: "Fries" }]);
  });
  it("handles fractional weights", () => {
    expect(parseItems("Pizza (x0.5)")).toEqual([{ name: "Pizza", share: "×0.5" }]);
  });
  it("shows sentences from a percentage or amount split as one chip", () => {
    expect(parseItems("20% of the bill")).toEqual([{ name: "20% of the bill" }]);
  });
  it("is empty when nothing is assigned", () => {
    expect(parseItems("(nothing assigned)")).toEqual([]);
    expect(parseItems("(not part of this split)")).toEqual([]);
    expect(parseItems("")).toEqual([]);
    expect(parseItems(undefined)).toEqual([]);
  });
});
