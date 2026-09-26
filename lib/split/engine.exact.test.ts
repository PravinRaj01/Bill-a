import { describe, expect, it } from "vitest";
import { computeSplit } from "./engine";
import type { AssignmentPlan, Receipt } from "@/types/domain";

const receipt = (o: Partial<Receipt> = {}): Receipt => ({ items: [], tax: 0, total: 0, currency: "RM", ...o });
const plan = (o: Partial<AssignmentPlan> = {}): AssignmentPlan => ({ assignments: [], defaultRule: "equal", notes: "", ...o });
const totalOf = (splits: { amount: number }[]) => splits.reduce((a, s) => a + s.amount, 0);

// The Real Food receipt from the field: two items + 6% GST = RM37.00
const real = receipt({
  items: [
    { name: "MUSH NOODLES DRY", quantity: 1, unitPrice: 1887, totalPrice: 1887 },
    { name: "STEAM DUMPLINGS", quantity: 1, unitPrice: 1604, totalPrice: 1604 },
  ],
  tax: 209,
  total: 3700,
});

describe("computeSplit: sum, divide, then round ONCE (no per-item cent bias)", () => {
  it("splits RM37.00 between two people as 18.50 / 18.50, not 18.51 / 18.49", () => {
    const r = computeSplit(real, ["Pravin", "Wifey"], plan(), true);
    expect(r.splits.map((s) => s.amount)).toEqual([1850, 1850]);
  });

  it("gives every person their exact share to within one cent, whatever the odd cents", () => {
    const items = [1000, 1000, 1000, 1001].map((c, i) => ({ name: `I${i}`, quantity: 1, unitPrice: c, totalPrice: c }));
    const r = computeSplit(receipt({ items, total: 4001 }), ["A", "B", "C"], plan(), false);
    const exact = 4001 / 3;
    for (const s of r.splits) expect(Math.abs(s.amount - exact)).toBeLessThan(1);
    expect(totalOf(r.splits)).toBe(4001);
  });

  it("an odd total between two people differs by exactly one cent, no more", () => {
    const r = computeSplit(receipt({ items: [{ name: "X", quantity: 1, unitPrice: 3701, totalPrice: 3701 }], total: 3701 }), ["A", "B"], plan(), false);
    expect(r.splits.map((s) => s.amount).sort((a, b) => a - b)).toEqual([1850, 1851]);
  });

  it("many odd items shared by two: the two totals never differ by more than a cent", () => {
    const items = Array.from({ length: 25 }, (_, i) => ({ name: `I${i}`, quantity: 1, unitPrice: 101 + 2 * i, totalPrice: 101 + 2 * i }));
    const total = items.reduce((s, it) => s + it.totalPrice, 0);
    const r = computeSplit(receipt({ items, total }), ["A", "B"], plan(), false);
    expect(Math.abs(r.splits[0].amount - r.splits[1].amount)).toBeLessThanOrEqual(1);
  });

  it("weighted shares are still exact (2:1 of 10.00 -> 6.67 / 3.33)", () => {
    const r = computeSplit(
      receipt({ items: [{ name: "X", quantity: 1, unitPrice: 1000, totalPrice: 1000 }], total: 1000 }),
      ["A", "B"],
      plan({ assignments: [{ itemIndex: 0, people: ["A", "B"], weights: [2, 1] }] }),
      false,
    );
    expect(r.splits.map((s) => s.amount)).toEqual([667, 333]);
  });

  it("fractional weights are handled exactly too (no float drift)", () => {
    const r = computeSplit(
      receipt({ items: [{ name: "X", quantity: 1, unitPrice: 1000, totalPrice: 1000 }], total: 1000 }),
      ["A", "B"],
      plan({ assignments: [{ itemIndex: 0, people: ["A", "B"], weights: [0.5, 1.5] }] }),
      false,
    );
    expect(r.splits.map((s) => s.amount)).toEqual([250, 750]);
  });

  it("the reasoning log explains that it rounds once, at the end", () => {
    const r = computeSplit(real, ["Pravin", "Wifey"], plan(), true);
    expect(r.reasoning).toMatch(/rounded once, at the end/);
    expect(r.reasoning).toMatch(/Pravin 18\.50, Wifey 18\.50/);
    expect(r.reasoning).toMatch(/Reconciled: 37\.00 === receipt 37\.00/);
  });

  it("handles a discount line and mixed groups without breaking the invariant", () => {
    const items = [
      { name: "A", quantity: 1, unitPrice: 1000, totalPrice: 1000 },
      { name: "PROMO", quantity: 1, unitPrice: -333, totalPrice: -333 },
      { name: "B", quantity: 1, unitPrice: 777, totalPrice: 777 },
    ];
    const r = computeSplit(receipt({ items, total: 1444 }), ["X", "Y", "Z"], plan({ assignments: [{ itemIndex: 2, people: ["X", "Z"] }] }), false);
    expect(totalOf(r.splits)).toBe(1444);
  });

  it("property: across random bills the total is exact and nobody is more than a cent from their exact share", () => {
    // Deterministic pseudo-random generator so a failure is reproducible.
    let seed = 12345;
    const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
    for (let n = 0; n < 300; n++) {
      const people = Array.from({ length: 2 + Math.floor(rnd() * 5) }, (_, i) => `P${i}`);
      const items = Array.from({ length: 1 + Math.floor(rnd() * 8) }, (_, i) => {
        const c = 50 + Math.floor(rnd() * 5000);
        return { name: `I${i}`, quantity: 1, unitPrice: c, totalPrice: c };
      });
      const subtotal = items.reduce((s, it) => s + it.totalPrice, 0);
      const total = Math.round(subtotal * (1 + rnd() * 0.16));
      const r = computeSplit(receipt({ items, total }), people, plan(), true);
      expect(totalOf(r.splits)).toBe(total);
      // everyone shares every item equally, so everyone's exact share is total/people
      for (const s of r.splits) expect(Math.abs(s.amount - total / people.length)).toBeLessThan(1);
    }
  });
});

describe("computeSplit: who pays the tax / service charge", () => {
  const allToPravin = plan({ assignments: [{ itemIndex: 0, people: ["Pravin"] }, { itemIndex: 1, people: ["Pravin"] }] });

  it("'Pravin pays the food, Wifey pays the tax': 34.91 / 2.09", () => {
    const r = computeSplit(real, ["Pravin", "Wifey"], { ...allToPravin, taxPayers: ["Wifey"] }, true);
    expect(r.splits.map((s) => s.amount)).toEqual([3491, 209]);
    expect(r.reasoning).toMatch(/paid by Wifey as instructed/);
    expect(r.splits[1].items).toBe("tax/service");
  });

  it("without a tax instruction the tax follows the food (the default): all on Pravin", () => {
    const r = computeSplit(real, ["Pravin", "Wifey"], allToPravin, true);
    expect(r.splits.map((s) => s.amount)).toEqual([3700, 0]);
  });

  it("several tax payers share it equally", () => {
    const r = computeSplit(real, ["Pravin", "Wifey", "Sam"], { ...allToPravin, taxPayers: ["Wifey", "Sam"] }, true);
    expect(r.splits.map((s) => s.amount)).toEqual([3491, 105, 104]); // 2.09 / 2, the odd cent to the first payer
    expect(totalOf(r.splits)).toBe(3700);
  });

  it("is ignored when tax is switched off", () => {
    const r = computeSplit(real, ["Pravin", "Wifey"], { ...allToPravin, taxPayers: ["Wifey"] }, false);
    expect(r.splits.map((s) => s.amount)).toEqual([3491, 0]);
  });

  it("ignores names that aren't in the bill, and falls back to the default if none remain", () => {
    const r = computeSplit(real, ["Pravin", "Wifey"], { ...allToPravin, taxPayers: ["Mallory"] }, true);
    expect(r.splits.map((s) => s.amount)).toEqual([3700, 0]);
  });

  it("a tax payer who also ate pays their food plus the whole tax", () => {
    const r = computeSplit(real, ["Pravin", "Wifey"], { ...plan(), taxPayers: ["Wifey"] }, true);
    expect(totalOf(r.splits)).toBe(3700);
    // food is 34.91 shared equally (17.455 each); Wifey adds all 2.09 => exactly 17.455 vs 19.545.
    // Both have a half-cent remainder (a tie), so the one leftover cent goes to the first person.
    expect(r.splits.map((s) => s.amount)).toEqual([1746, 1954]);
  });

  it("with excluded items, the tax payer pays only the tax on the covered portion", () => {
    const r = computeSplit(
      real,
      ["Pravin", "Wifey"],
      { assignments: [{ itemIndex: 0, people: ["Pravin"] }], defaultRule: "exclude", notes: "", taxPayers: ["Wifey"] },
      true,
    );
    expect(r.splits[0].amount).toBe(1887);
    expect(r.splits[1].amount).toBe(113); // 2.09 * 18.87 / 34.91
  });

  it("property: tax payers never break reconciliation", () => {
    let seed = 777;
    const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
    for (let n = 0; n < 200; n++) {
      const people = Array.from({ length: 2 + Math.floor(rnd() * 4) }, (_, i) => `P${i}`);
      const items = Array.from({ length: 1 + Math.floor(rnd() * 6) }, (_, i) => {
        const c = 50 + Math.floor(rnd() * 4000);
        return { name: `I${i}`, quantity: 1, unitPrice: c, totalPrice: c };
      });
      const subtotal = items.reduce((s, it) => s + it.totalPrice, 0);
      const total = Math.round(subtotal * (1 + rnd() * 0.2));
      const payers = people.filter(() => rnd() < 0.5);
      const r = computeSplit(receipt({ items, total }), people, plan({ taxPayers: payers }), true);
      expect(totalOf(r.splits)).toBe(total);
    }
  });
});
