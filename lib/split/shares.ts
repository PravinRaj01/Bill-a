import { SplitReconciliationError, type Cents, type SplitRecord, type SplitResult } from "@/types/domain";

// Whole-bill shares ("Alexis 70%, Riley 30%", "Riley pays RM30, the rest equally"). Unlike an
// item split there is nothing to look up on the receipt: the shares are just numbers the user
// gave, so this is plain arithmetic that never touches a model. parseShareInstruction() turns
// the words into per-person amounts (via apportion, which reconciles by construction); this
// builds the same SplitResult shape computeSplit() returns, with a reasoning log, and asserts
// the amounts add up to the bill.

const fmt = (c: Cents) => (c / 100).toFixed(2);

export function computeShareSplit(
  people: string[],
  amounts: Cents[],
  labels: string[],
  total: Cents,
  preface: string[] = [],
): SplitResult {
  if (people.length === 0 || amounts.length !== people.length) {
    throw new Error("computeShareSplit: people and amounts must line up");
  }
  const actual = amounts.reduce((a, b) => a + b, 0);
  if (actual !== total) throw new SplitReconciliationError(total, actual);

  const splits: SplitRecord[] = people.map((name, i) => ({
    name,
    amount: amounts[i],
    items: labels[i] || "(not part of this split)",
  }));

  const trace = [
    ...preface,
    ...people.map((p, i) => `${p}: ${fmt(amounts[i])}${labels[i] ? ` — ${labels[i]}` : ""}`),
    `✓ Reconciled: ${fmt(actual)} === bill ${fmt(total)}`,
  ];
  return { splits, reasoning: trace.join("\n"), verified: true };
}
