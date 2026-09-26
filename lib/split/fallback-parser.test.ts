import { describe, expect, it } from "vitest";
import { BAKEOFF_CASES } from "@/lib/ai/bakeoff-cases";
import { HELDOUT_CASES } from "./heldout-cases";
import { evaluateFallback, summarise } from "./fallback-eval";
import { parseInstruction, splitClauses } from "./fallback-parser";

// MEASURED accuracy of the no-LLM path (recorded here so a regression is loud).
//
//   Development set (the 15 bake-off cases; the parser was built against these,
//   so this number is optimistic):   14 correct · 1 asks a question · 0 wrong
//   Held-out set (20 new phrasings, written before the parser existed):
//                                    16 correct · 4 ask a question · 0 wrong
//   (First held-out run, before one general fix to "Everyone except <person> …":
//    15 correct · 4 ask · 1 wrong.)
//
// "Asks a question" = the parser returned a chip instead of a plan it wasn't
// sure of; the one dev-set case is "Sarah had the tea" against an item named
// TEH TARIK (a synonym the lexicon doesn't bridge). The invariant to protect is
// ZERO confidently-wrong plans.

describe("fallback parser — measured accuracy", () => {
  const dev = BAKEOFF_CASES.map(evaluateFallback);
  const held = HELDOUT_CASES.map(evaluateFallback);

  it("dev set: never confidently wrong, at least 14/15 fully correct", () => {
    const s = summarise(dev);
    expect(s.wrong, JSON.stringify(dev.filter((o) => o.verdict === "wrong"))).toBe(0);
    expect(s.correct).toBeGreaterThanOrEqual(14);
  });

  it("held-out set: never confidently wrong, at least 16/20 fully correct", () => {
    const s = summarise(held);
    expect(s.wrong, JSON.stringify(held.filter((o) => o.verdict === "wrong"))).toBe(0);
    expect(s.correct).toBeGreaterThanOrEqual(16);
  });
});

const people = ["Pravin", "Aisha", "Sarah"];
const items = [
  { name: "FRIED RICE", quantity: 1, unitPrice: 900, totalPrice: 900 },
  { name: "CHICKEN RENDANG", quantity: 1, unitPrice: 1300, totalPrice: 1300 },
  { name: "TEH TARIK", quantity: 1, unitPrice: 400, totalPrice: 400 },
  { name: "ICED LEMON TEA", quantity: 1, unitPrice: 400, totalPrice: 400 },
  { name: "MILKSHAKE", quantity: 1, unitPrice: 600, totalPrice: 600 },
];

describe("splitClauses", () => {
  it("keeps lists together and splits real clauses", () => {
    expect(splitClauses("Split the pizza between Pravin and Aisha only, everyone splits the rest", people)).toEqual([
      "Split the pizza between Pravin and Aisha only",
      "everyone splits the rest",
    ]);
    expect(splitClauses("Pravin had the rice and Aisha had the chicken", people)).toEqual([
      "Pravin had the rice",
      "Aisha had the chicken",
    ]);
    expect(splitClauses("Pravin and Aisha split the rice and chicken", people)).toEqual([
      "Pravin and Aisha split the rice and chicken",
    ]);
    expect(splitClauses("Everyone but Sarah splits the milkshake", people)).toHaveLength(1);
  });
});

describe("parseInstruction — never guesses", () => {
  it("asks about a typo'd name instead of resolving it", () => {
    const r = parseInstruction("Pravn had the rice", people, items);
    expect(r.chips).toContainEqual({ kind: "possible-typo", token: "Pravn", suggestion: "Pravin" });
    expect(r.plan.assignments).toEqual([]); // nothing assigned to a guessed person
  });

  it("asks which item when a phrase ties between several", () => {
    const tie = parseInstruction("Sarah had the fried", people, [
      { name: "FRIED RICE", quantity: 1, unitPrice: 1, totalPrice: 1 },
      { name: "FRIED NOODLE", quantity: 1, unitPrice: 1, totalPrice: 1 },
    ]);
    expect(tie.chips.some((c) => c.kind === "ambiguous-item")).toBe(true);
    expect(tie.plan.assignments).toEqual([]);
  });

  it("leaves an item carved out of 'everyone' as a question when nobody is given it", () => {
    const r = parseInstruction("Everyone splits equally except the milkshake", people, items);
    expect(r.chips.some((c) => c.kind === "exception-unassigned")).toBe(true);
    expect(r.plan.assignments).toEqual([]);
  });

  it("asks about a person who isn't in the group", () => {
    const r = parseInstruction("Wei owes for the rice", people, items);
    expect(r.chips).toContainEqual({ kind: "unknown-person", token: "Wei" });
  });

  it("gives a plain instruction an empty, valid plan", () => {
    const r = parseInstruction("Split everything equally", people, items);
    expect(r.chips).toEqual([]);
    expect(r.plan).toMatchObject({ assignments: [], defaultRule: "equal" });
  });

  it("handles the milkshake carve-out end to end", () => {
    const r = parseInstruction("Everyone splits equally except the milkshake, which is just for Sarah", people, items);
    expect(r.plan.assignments).toEqual([{ itemIndex: 4, people: ["Sarah"] }]);
    expect(r.chips).toEqual([]);
  });
});
