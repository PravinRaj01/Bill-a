import type { Cents } from "@/types/domain";

/**
 * Distributes an integer total across weighted shares using Hamilton's
 * largest-remainder method, with zero residue BY CONSTRUCTION.
 *
 * This replaces the old prompt's "adjust pennies on the first person if
 * needed" instruction, which systematically overcharged whoever was
 * listed first. Every share here is computed independently and the
 * leftover cents (there are at most `weights.length - 1` of them, from
 * the floor() truncation) go to the shares with the largest fractional
 * remainder — the textbook fair-division algorithm for indivisible units.
 *
 * Invariant, always: `apportion(total, weights).reduce((a,b) => a+b) === total`.
 * This is what lets computeSplit() assert reconciliation instead of hoping
 * for it.
 */
export function apportion(total: Cents, weights: number[]): Cents[] {
  if (weights.length === 0) return [];

  const sum = weights.reduce((a, b) => a + b, 0);
  if (sum <= 0) {
    // No one has a positive weight — nothing to distribute. Give the
    // remainder to the first entry rather than silently dropping cents,
    // so the caller's invariant still holds even in this degenerate case.
    const out = weights.map(() => 0);
    out[0] = total;
    return out;
  }

  const exact = weights.map((w) => (total * w) / sum);
  const floors = exact.map(Math.floor);
  const distributed = floors.reduce((a, b) => a + b, 0);
  let residue = total - distributed;

  // Largest fractional remainder gets the leftover cents first. Ties
  // broken by index so the result is deterministic given the same input.
  const order = exact
    .map((e, i) => ({ i, frac: e - Math.floor(e) }))
    .sort((a, b) => b.frac - a.frac || a.i - b.i);

  const out = [...floors];
  for (let k = 0; residue > 0; k++, residue--) {
    out[order[k % out.length].i]++;
  }
  return out;
}
