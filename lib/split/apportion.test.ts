import { describe, expect, it } from "vitest";
import { apportion } from "./apportion";

function sum(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0);
}

describe("apportion", () => {
  it("splits an indivisible total evenly, handing leftover cents to the largest remainders", () => {
    // RM 10.00 / 3 people — the textbook indivisible case.
    const shares = apportion(1000, [1, 1, 1]);
    expect(sum(shares)).toBe(1000);
    // 1000/3 = 333.33..., so two people get 334 and one gets 333 (or some
    // permutation) rather than all the leftover landing on person 0.
    expect(shares.sort((a, b) => a - b)).toEqual([333, 333, 334]);
  });

  it("gives the whole amount to a single weight", () => {
    expect(apportion(1234, [1])).toEqual([1234]);
  });

  it("respects unequal weights proportionally", () => {
    // 3:1 split of 1000 -> exact would be 750/250, no remainder.
    expect(apportion(1000, [3, 1])).toEqual([750, 250]);
  });

  it("distributes remainders by largest fractional part, not by index order", () => {
    // weights [1,1,1,1,1] over 1003 cents: each exact share is 200.6,
    // floor 200 each (1000 total), 3 cents left over, all fractional
    // parts tied at .6 -> tie-break by index gives the first three.
    const shares = apportion(1003, [1, 1, 1, 1, 1]);
    expect(sum(shares)).toBe(1003);
    expect(shares).toEqual([201, 201, 201, 200, 200]);
  });

  it("handles a zero total", () => {
    expect(apportion(0, [1, 1, 1])).toEqual([0, 0, 0]);
  });

  it("handles a single-element weights array with a large total", () => {
    expect(apportion(999999, [1])).toEqual([999999]);
  });

  it("never drops or fabricates cents regardless of weight skew", () => {
    const shares = apportion(101, [1, 2, 3, 97]);
    expect(sum(shares)).toBe(101);
  });

  it("handles negative totals (discount lines) without losing the invariant", () => {
    const shares = apportion(-150, [1, 1, 1]);
    expect(sum(shares)).toBe(-150);
  });

  it("falls back to giving everything to the first entry when all weights are zero", () => {
    // Degenerate case: nobody has positive weight. Rather than silently
    // dropping the total (which would break every caller's invariant),
    // apportion() assigns it to index 0.
    const shares = apportion(500, [0, 0, 0]);
    expect(sum(shares)).toBe(500);
  });

  it("returns an empty array for an empty weights list", () => {
    expect(apportion(500, [])).toEqual([]);
  });

  it("holds the sum invariant across many random weight vectors and totals", () => {
    // Deterministic PRNG so failures are reproducible without adding a
    // property-testing dependency for one invariant check.
    let seed = 42;
    const rand = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff;
    };

    for (let trial = 0; trial < 2000; trial++) {
      const n = 1 + Math.floor(rand() * 8);
      const weights = Array.from({ length: n }, () => 1 + Math.floor(rand() * 50));
      const total = Math.floor(rand() * 100000);

      const shares = apportion(total, weights);
      expect(sum(shares)).toBe(total);
      expect(shares.length).toBe(n);
    }
  });
});
