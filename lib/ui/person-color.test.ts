import { describe, expect, it } from "vitest";
import { initials, personColorIndex, PERSON_COLOR_COUNT } from "./person-color";

describe("personColorIndex", () => {
  it("is stable and ignores case and surrounding spaces", () => {
    expect(personColorIndex("Pravin")).toBe(personColorIndex("  pravin "));
  });
  it("always lands in 1..8", () => {
    for (const n of ["a", "Lakx", "Wei Jie", "மதி", "", "🍜"]) {
      const i = personColorIndex(n);
      expect(i).toBeGreaterThanOrEqual(1);
      expect(i).toBeLessThanOrEqual(PERSON_COLOR_COUNT);
    }
  });
});

describe("initials", () => {
  it("takes first+last for multi-word names, first two letters otherwise", () => {
    expect(initials("Pravin Raj")).toBe("PR");
    expect(initials("Ali bin Abu")).toBe("AA");
    expect(initials("lakx")).toBe("LA");
    expect(initials("  ")).toBe("?");
  });
});
