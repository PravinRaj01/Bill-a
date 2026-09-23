import { describe, expect, it } from "vitest";
import { computeSplit } from "@/lib/split/engine";
import { BAKEOFF_CASES } from "./bakeoff-cases";

// Sanity-checks the bake-off fixture data itself, independent of any
// model: every hand-authored expectedPlan must actually reconcile
// through the real engine before it's trusted as ground truth for
// scoring model output. This is what stops a typo in bakeoff-cases.ts
// from silently producing a bogus "0% accuracy" or "100% accuracy" for
// every model.
describe("bakeoff cases are internally valid", () => {
  it.each(BAKEOFF_CASES.map((c) => [c.id, c] as const))(
    "%s: expectedPlan reconciles without throwing, and covers every named person",
    (_id, c) => {
      const result = computeSplit(c.receipt, c.people, c.expectedPlan, c.applyTax);
      expect(result.splits).toHaveLength(c.people.length);
      const total = result.splits.reduce((s, r) => s + r.amount, 0);
      expect(total).toBeGreaterThanOrEqual(0);
    },
  );

  it("every case has a non-empty instruction and at least one item", () => {
    for (const c of BAKEOFF_CASES) {
      expect(c.instruction.length).toBeGreaterThan(0);
      expect(c.receipt.items.length).toBeGreaterThan(0);
      expect(c.people.length).toBeGreaterThan(0);
    }
  });

  it("has unique case ids", () => {
    const ids = BAKEOFF_CASES.map((c) => c.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
});
