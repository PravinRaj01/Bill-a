import { computeSplit } from "@/lib/split/engine";
import type { AssignmentPlan, Receipt, SplitResult } from "@/types/domain";
import { resolveInstruction } from "@/lib/retrieval/resolve";
import { parseInstruction, type Chip } from "@/lib/split/fallback-parser";
import { findAmbiguities, routeIntent } from "./intentRouter";
import { runCascade, type Attempt, type CascadeOptions } from "./providers/cascade";
import type { ProviderId } from "./providers/types";

// The whole "instruction -> split" pipeline, in one place:
//
//   route (regex, <1 ms)  ->  ambiguity check  ->  resolve references
//     ->  cascade (Groq -> Gemini -> fallback)  ->  validatePlan  ->  computeSplit
//     ->  cross-check against the on-device rules
//
// The LLM only ever produces an AssignmentPlan; every cent comes from computeSplit.
//
// Cross-check: a cloud model can be confidently wrong, and a wrong split looks exactly like a
// right one (measured live: each model missed cases the on-device rules get right, and the
// misses don't overlap; one model even flipped between right and wrong across identical calls).
// So when the model's split and the rules' split disagree, and the rules genuinely understood
// the instruction, the user is shown both and picks. No extra API calls: the rules are local.

export interface PlanSplitInput {
  receipt: Receipt;
  people: string[];
  /** The original instruction, then any follow-up modifications, in order. */
  instructions: string[];
  applyTax: boolean;
  keys: CascadeOptions["keys"];
  /** Skip the "did you mean…?" gate (the user chose "continue anyway"). */
  ignoreAmbiguities?: boolean;
  timeoutMs?: number;
  fetchImpl?: typeof fetch;
  models?: CascadeOptions["models"];
  /** Compare a cloud answer with the on-device rules (default true). Off for benchmarks that score the model alone. */
  crossCheck?: boolean;
}

export type PlanOutcome =
  | { kind: "add-members"; names: string[] }
  | { kind: "needs-clarification"; chips: Chip[]; /** best guess, if the user wants to go ahead */ preview?: SplitPreview }
  | {
      kind: "split";
      result: SplitResult;
      plan: AssignmentPlan;
      tier: ProviderId | "fallback";
      attempts: Attempt[];
      repairs: string[];
    }
  | {
      /** The cloud model and the on-device rules gave different splits; the user chooses. */
      kind: "disagreement";
      ai: SplitChoice & { tier: ProviderId; attempts: Attempt[]; repairs: string[] };
      rules: SplitChoice;
    };

/** One side of a cross-check disagreement. */
export interface SplitChoice {
  result: SplitResult;
  plan: AssignmentPlan;
}

export interface SplitPreview {
  result: SplitResult;
  plan: AssignmentPlan;
}

export async function planSplit(input: PlanSplitInput): Promise<PlanOutcome> {
  const { receipt, people } = input;
  const instructions = input.instructions.map((t) => t.trim()).filter(Boolean);
  if (instructions.length === 0) instructions.push("Split equally");
  const joined = instructions.join(". ");
  const latest = instructions[instructions.length - 1];

  if (people.length === 0) throw new Error("planSplit: no people");
  if (receipt.items.length === 0) throw new Error("planSplit: no items");

  // 1. "add Wei" is a command for the people list, not a split.
  const route = routeIntent(latest, people, receipt.items);
  if (route.intent === "ADD_MEMBERS") return { kind: "add-members", names: route.newNames };

  // 2. Ask about typos / unknown names / tied item phrases BEFORE spending an LLM call.
  if (!input.ignoreAmbiguities) {
    const ambiguities = findAmbiguities(joined, people, receipt.items);
    if (ambiguities.length > 0) return { kind: "needs-clarification", chips: ambiguities };
  }

  // 3. Resolve "the drinks" / "the rest" in code, then let the cascade build the plan.
  const resolution = resolveInstruction(joined, receipt.items);
  const cascade = await runCascade(
    { people, items: receipt.items, instructions, resolvedBlock: resolution.promptBlock },
    { keys: input.keys, timeoutMs: input.timeoutMs, fetchImpl: input.fetchImpl, models: input.models },
  );

  // 4. All the arithmetic, in integer cents.
  const result = computeSplit(receipt, people, cascade.plan, input.applyTax);

  // The fallback parser asks questions instead of guessing; surface them, with its
  // best guess attached in case the user just wants to go ahead.
  if (cascade.tier === "fallback" && cascade.chips.length > 0) {
    return { kind: "needs-clarification", chips: cascade.chips, preview: { result, plan: cascade.plan } };
  }

  // 5. Cross-check a cloud answer against the local rules.
  if (cascade.tier !== "fallback" && input.crossCheck !== false) {
    const rules = parseInstruction(joined, people, receipt.items);
    if (rules.understood && rules.chips.length === 0) {
      const rulesResult = computeSplit(receipt, people, rules.plan, input.applyTax);
      const differs = rulesResult.splits.some((s, i) => s.amount !== result.splits[i].amount);
      if (differs) {
        return {
          kind: "disagreement",
          ai: { result, plan: cascade.plan, tier: cascade.tier, attempts: cascade.attempts, repairs: cascade.repairs },
          rules: { result: rulesResult, plan: rules.plan },
        };
      }
    }
  }

  return {
    kind: "split",
    result,
    plan: cascade.plan,
    tier: cascade.tier,
    attempts: cascade.attempts,
    repairs: cascade.repairs,
  };
}
