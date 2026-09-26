import { validatePlan } from "@/lib/ai/validatePlan";
import { computeSplit } from "@/lib/split/engine";
import { parseInstruction } from "@/lib/split/fallback-parser";
import type { BakeoffCase } from "@/lib/ai/bakeoff-cases";

export type Verdict = "correct" | "safe-chip" | "wrong";

export interface CaseOutcome {
  id: string;
  verdict: Verdict;
  detail: string;
}

const amounts = (c: BakeoffCase, plan: BakeoffCase["expectedPlan"]) =>
  computeSplit(c.receipt, c.people, plan, c.applyTax).splits.map((s) => s.amount);

/**
 * Same yardstick as the model bake-off: run BOTH the expected plan and the
 * parser's plan through the real engine and compare per-person cents. On top of
 * pass/fail, a case the parser answered with a question chip instead of a
 * confident plan counts as "safe-chip" — asking is acceptable, guessing wrong is not.
 */
export function evaluateFallback(c: BakeoffCase & { saferAsChip?: boolean; mustAsk?: boolean }): CaseOutcome {
  const { plan, chips } = parseInstruction(c.instruction, c.people, c.receipt.items);
  const checked = validatePlan(plan, { itemCount: c.receipt.items.length, people: c.people });
  if (!checked.ok) {
    return { id: c.id, verdict: chips.length > 0 ? "safe-chip" : "wrong", detail: `invalid plan: ${checked.reason}` };
  }
  if (c.mustAsk) {
    return { id: c.id, verdict: chips.length > 0 ? "safe-chip" : "wrong", detail: chips.length > 0 ? "asked, as expected" : "expected a question, got a confident plan" };
  }
  const want = amounts(c, c.expectedPlan);
  const got = amounts(c, checked.plan);
  if (want.every((v, i) => v === got[i])) return { id: c.id, verdict: "correct", detail: "" };
  const detail = `want ${want.join("/")} got ${got.join("/")}; chips=${chips.map((x) => x.kind).join(",") || "none"}`;
  return { id: c.id, verdict: chips.length > 0 ? "safe-chip" : "wrong", detail };
}

export function summarise(outcomes: CaseOutcome[]) {
  const n = (v: Verdict) => outcomes.filter((o) => o.verdict === v).length;
  return { total: outcomes.length, correct: n("correct"), safeChip: n("safe-chip"), wrong: n("wrong") };
}
