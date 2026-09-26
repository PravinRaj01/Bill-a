import { describe, expect, it } from "vitest";
import { validatePlan } from "./validatePlan";

const ctx = { itemCount: 3, people: ["Pravin", "Aisha", "Sarah"] };
const plan = (assignments: unknown[], defaultRule = "equal") => ({ assignments, defaultRule, notes: "" });

describe("validatePlan", () => {
  it("accepts a clean plan unchanged", () => {
    const r = validatePlan(plan([{ itemIndex: 1, people: ["Pravin"] }]), ctx);
    expect(r).toEqual({
      ok: true,
      repairs: [],
      plan: { assignments: [{ itemIndex: 1, people: ["Pravin"] }], defaultRule: "equal", notes: "" },
    });
  });

  it("accepts a JSON string (what the providers return)", () => {
    expect(validatePlan(JSON.stringify(plan([])), ctx).ok).toBe(true);
  });

  it.each([
    ["malformed JSON", "{not json"],
    ["not an object", 42],
    ["missing defaultRule", { assignments: [] }],
    ["bad defaultRule", plan([], "sometimes")],
    ["assignments not an array", { assignments: "x", defaultRule: "equal" }],
    ["non-integer itemIndex", plan([{ itemIndex: 1.5, people: ["Pravin"] }])],
  ])("rejects %s", (_n, raw) => {
    expect(validatePlan(raw, ctx).ok).toBe(false);
  });

  it("rejects an out-of-range item index (both ends)", () => {
    expect(validatePlan(plan([{ itemIndex: 3, people: ["Pravin"] }]), ctx).ok).toBe(false);
    expect(validatePlan(plan([{ itemIndex: -1, people: ["Pravin"] }]), ctx).ok).toBe(false);
  });

  it("drops unknown names but keeps the rest, and canonicalises case", () => {
    const r = validatePlan(plan([{ itemIndex: 0, people: ["pravin", "Mallory"] }]), ctx);
    expect(r).toMatchObject({ ok: true, plan: { assignments: [{ itemIndex: 0, people: ["Pravin"] }] } });
  });

  it("rejects an assignment whose names are all unknown (would silently change who pays)", () => {
    expect(validatePlan(plan([{ itemIndex: 0, people: ["Mallory"] }]), ctx).ok).toBe(false);
    expect(validatePlan(plan([{ itemIndex: 0, people: [] }]), ctx).ok).toBe(false);
  });

  it("dedupes names within an assignment", () => {
    const r = validatePlan(plan([{ itemIndex: 0, people: ["Aisha", "aisha", "Aisha"] }]), ctx);
    expect(r).toMatchObject({ ok: true, plan: { assignments: [{ itemIndex: 0, people: ["Aisha"] }] } });
  });

  it("dedupes itemIndex across assignments — last one wins, like computeSplit", () => {
    const r = validatePlan(
      plan([{ itemIndex: 2, people: ["Pravin"] }, { itemIndex: 2, people: ["Sarah"] }]),
      ctx,
    );
    expect(r).toMatchObject({ ok: true, plan: { assignments: [{ itemIndex: 2, people: ["Sarah"] }] } });
  });

  it("drops weights that no longer line up with the surviving names", () => {
    const r = validatePlan(plan([{ itemIndex: 0, people: ["Pravin", "Mallory"], weights: [1, 2] }]), ctx);
    expect(r).toMatchObject({ ok: true, plan: { assignments: [{ itemIndex: 0, people: ["Pravin"] }] } });
    expect((r as { plan: { assignments: object[] } }).plan.assignments[0]).not.toHaveProperty("weights");
  });
});
