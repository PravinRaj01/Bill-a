import type { Cents, SplitRecord } from "@/types/domain";

// Cash-friendly shares. computeSplit() returns exact-to-the-cent shares; when people settle in
// cash they'd rather owe 54.50 than 54.47. Rounding every share independently would no
// longer add up to the bill, so ONE person (the "absorber") takes whatever is left over:
// everyone else is rounded to the nearest `unit`, and the absorber pays total - the rest.
// The total is therefore preserved exactly, by construction.

/** The units offered in the UI, in cents: 1 = exact, 5 = nearest 5 sen, 100 = nearest RM1. */
export const CASH_UNITS = [1, 5, 100] as const;
export type CashUnit = (typeof CASH_UNITS)[number];

export interface CashRounded {
  amounts: Cents[];
  /** Per person: rounded minus exact (what the rounding did to them). Sums to 0. */
  deltas: Cents[];
  /** Index of the person who absorbed the difference. */
  absorber: number;
}

/** The person who absorbs by default: whoever pays the most (a few sen matter least to them). */
export function defaultAbsorber(amounts: Cents[]): number {
  let best = 0;
  amounts.forEach((a, i) => {
    if (a > amounts[best]) best = i;
  });
  return best;
}

export function applyCashRounding(amounts: Cents[], unit: number, absorber = defaultAbsorber(amounts)): CashRounded {
  const exact: CashRounded = { amounts: [...amounts], deltas: amounts.map(() => 0), absorber };
  if (unit <= 1 || amounts.length < 2 || absorber < 0 || absorber >= amounts.length) return exact;

  const total = amounts.reduce((a, b) => a + b, 0);
  // Nobody who owes something is rounded down to owing nothing: at least one unit.
  const rounded = amounts.map((a, i) => (i === absorber ? a : a > 0 ? Math.max(unit, Math.round(a / unit) * unit) : 0));
  rounded[absorber] = total - rounded.reduce((sum, v, i) => (i === absorber ? sum : sum + v), 0);
  // Small shares next to a big rounding unit could leave the absorber with a negative share: refuse then.
  if (rounded[absorber] < 0) return exact;

  return { amounts: rounded, deltas: rounded.map((v, i) => v - amounts[i]), absorber };
}

/** applyCashRounding over named shares: the absorber is picked by name (default: the biggest payer). */
export function roundSplit(
  exact: SplitRecord[],
  unit: number,
  absorberName: string | null,
): { splits: SplitRecord[]; deltas: Cents[]; absorber: number; changed: boolean } {
  const named = absorberName ? exact.findIndex((r) => r.name === absorberName) : -1;
  const r = applyCashRounding(exact.map((s) => s.amount), unit, named >= 0 ? named : undefined);
  return {
    splits: exact.map((s, i) => ({ ...s, amount: r.amounts[i] })),
    deltas: r.deltas,
    absorber: r.absorber,
    changed: r.deltas.some((d) => d !== 0),
  };
}
