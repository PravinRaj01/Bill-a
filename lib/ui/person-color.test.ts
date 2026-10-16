import { describe, expect, it } from "vitest";
import { initials, personColorIndex, PERSON_COLOR_COUNT } from "./person-color";

describe("personColorIndex", () => {
  it("is stable and ignores case and surrounding spaces", () => {
    expect(personColorIndex("Alexis")).toBe(personColorIndex("  alexis "));
  });
  it("always lands in 1..8", () => {
    for (const n of ["a", "Riley", "Wei Jie", "மதி", "", "🍜"]) {
      const i = personColorIndex(n);
      expect(i).toBeGreaterThanOrEqual(1);
      expect(i).toBeLessThanOrEqual(PERSON_COLOR_COUNT);
    }
  });
});

describe("initials", () => {
  it("takes first+last for multi-word names, first two letters otherwise", () => {
    expect(initials("Alexis Reed")).toBe("AR");
    expect(initials("Ali bin Abu")).toBe("AA");
    expect(initials("riley")).toBe("RI");
    expect(initials("  ")).toBe("?");
  });
});
