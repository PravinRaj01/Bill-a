import { describe, expect, it } from "vitest";
import { applyCashRounding, defaultAbsorber, roundSplit } from "./cashRound";

const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);

describe("applyCashRounding", () => {
  it("leaves shares alone for the exact unit, one person, or a bad absorber", () => {
    expect(applyCashRounding([3333, 3333, 3334], 1).amounts).toEqual([3333, 3333, 3334]);
    expect(applyCashRounding([1000], 5).amounts).toEqual([1000]);
    expect(applyCashRounding([500, 500], 5, 9).amounts).toEqual([500, 500]);
  });

  it("rounds everyone but the absorber to 5 sen; the absorber takes the difference", () => {
    const r = applyCashRounding([3333, 3333, 3334], 5);
    expect(r.absorber).toBe(2); // pays the most
    expect(r.amounts).toEqual([3335, 3335, 3330]);
    expect(sum(r.amounts)).toBe(10000);
    expect(r.deltas).toEqual([2, 2, -4]);
  });

  it("rounds to the nearest RM1 and keeps the total", () => {
    const r = applyCashRounding([5475, 5474], 100);
    expect(r.amounts).toEqual([5449, 5500]);
    expect(sum(r.amounts)).toBe(10949);
  });

  it("uses the chosen absorber", () => {
    const r = applyCashRounding([3333, 3333, 3334], 5, 0);
    expect(r.amounts).toEqual([3330, 3335, 3335]);
    expect(r.absorber).toBe(0);
  });

  it("never lets the absorber go negative", () => {
    // 40 would round to 0 -> bumped to a full unit (100), leaving the absorber at -30: refused
    expect(applyCashRounding([30, 40], 100, 0).amounts).toEqual([30, 40]);
    // 60 -> 100, absorber -> 10 - 40 = -30: refused
    expect(applyCashRounding([60, 10], 100, 1).amounts).toEqual([60, 10]);
  });

  it("property: total preserved, non-absorbers are multiples of the unit, deltas sum to 0", () => {
    let seed = 7;
    const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
    for (let n = 0; n < 500; n++) {
      const people = 2 + Math.floor(rnd() * 6);
      const amounts = Array.from({ length: people }, () => Math.floor(rnd() * 20000));
      const unit = [5, 100][Math.floor(rnd() * 2)];
      const r = applyCashRounding(amounts, unit);
      expect(sum(r.amounts)).toBe(sum(amounts));
      expect(sum(r.deltas)).toBe(0);
      r.amounts.forEach((a, i) => {
        expect(a).toBeGreaterThanOrEqual(0);
        if (i !== r.absorber) expect(a % unit).toBe(0);
      });
    }
  });
});

describe("defaultAbsorber", () => {
  it("is the biggest payer, first on a tie", () => {
    expect(defaultAbsorber([100, 300, 300, 200])).toBe(1);
    expect(defaultAbsorber([5])).toBe(0);
  });
});

describe("roundSplit", () => {
  const exact = [
    { name: "Alexis", amount: 5475, items: "a" },
    { name: "Riley", amount: 5474, items: "b" },
  ];
  it("is a no-op (changed=false) for the exact unit", () => {
    const r = roundSplit(exact, 1, null);
    expect(r.splits).toEqual(exact);
    expect(r.changed).toBe(false);
  });
  it("rounds by name, keeping items, and reports the absorber's index", () => {
    const r = roundSplit(exact, 100, "Riley");
    expect(r.splits.map((s) => [s.name, s.amount, s.items])).toEqual([["Alexis", 5500, "a"], ["Riley", 5449, "b"]]);
    expect(r.absorber).toBe(1);
    expect(r.changed).toBe(true);
  });
  it("falls back to the biggest payer when the named absorber is gone", () => {
    expect(roundSplit(exact, 100, "Nobody").absorber).toBe(0);
  });
});
