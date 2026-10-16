import { describe, expect, it } from "vitest";
import { computeSplit } from "./engine";
import type { AssignmentPlan, Receipt } from "@/types/domain";

const receipt = (o: Partial<Receipt> = {}): Receipt => ({ items: [], tax: 0, total: 0, currency: "RM", ...o });
const plan = (o: Partial<AssignmentPlan> = {}): AssignmentPlan => ({ assignments: [], defaultRule: "equal", notes: "", ...o });
const sum = (splits: { amount: number }[]) => splits.reduce((a, s) => a + s.amount, 0);
const item = (name: string, c: number) => ({ name, quantity: 1, unitPrice: c, totalPrice: c });

describe("computeSplit with a printed rounding adjustment", () => {
  it("reconciles to the printed total when there is rounding and no tax line (the dropped-line bug)", () => {
    // 10.00 + 10.00 = 20.00, receipt rounds it down to 19.98
    const r = receipt({ items: [item("A", 1000), item("B", 1000)], total: 1998, rounding: -2 });
    const out = computeSplit(r, ["Alexis", "Riley"], plan(), true);
    expect(sum(out.splits)).toBe(1998);
    expect(out.splits.map((s) => s.amount)).toEqual([999, 999]);
    expect(out.reasoning).toMatch(/Rounding adjustment -0\.02/);
  });

  it("separates rounding from tax: 6% SST then a -0.02 rounding", () => {
    // 22.00 + 1.32 SST = 23.32, rounded to 23.30
    const r = receipt({ items: [item("A", 1500), item("B", 700)], tax: 132, total: 2330, rounding: -2 });
    const out = computeSplit(r, ["A", "B"], plan({ assignments: [{ itemIndex: 0, people: ["A"] }, { itemIndex: 1, people: ["B"] }] }), true);
    expect(sum(out.splits)).toBe(2330);
    // A's food 15.00 of 22.00 -> 15/22 of (22.00 + 1.32 - 0.02)
    expect(out.splits[0].amount).toBe(Math.round((1500 * 2330) / 2200));
  });

  it("a positive rounding adds to the bill", () => {
    const r = receipt({ items: [item("A", 1001), item("B", 1001)], total: 2004, rounding: 2 });
    expect(sum(computeSplit(r, ["x", "y"], plan(), true).splits)).toBe(2004);
  });

  it("ignores the rounding when tax & service is switched off (the split is the plain subtotal)", () => {
    const r = receipt({ items: [item("A", 1000), item("B", 1000)], total: 1998, rounding: -2 });
    expect(sum(computeSplit(r, ["x", "y"], plan(), false).splits)).toBe(2000);
  });

  it("is a no-op when rounding is absent or zero", () => {
    const base = receipt({ items: [item("A", 1887), item("B", 1604)], tax: 209, total: 3700 });
    const a = computeSplit(base, ["P", "W"], plan(), true);
    const b = computeSplit({ ...base, rounding: 0 }, ["P", "W"], plan(), true);
    expect(b.splits).toEqual(a.splits);
  });

  it("rounding is spread by what each person ordered while named tax payers still pay the tax", () => {
    const r = receipt({ items: [item("A", 3000), item("B", 1000)], tax: 400, total: 4398, rounding: -2 });
    const p = plan({ assignments: [{ itemIndex: 0, people: ["A"] }, { itemIndex: 1, people: ["B"] }], taxPayers: ["B"] });
    const out = computeSplit(r, ["A", "B"], p, true);
    expect(sum(out.splits)).toBe(4398);
    // B: food 10.00 + all 4.00 tax, minus 1/4 of the 0.02 rounding (rounds to 0.01 off A, 0.01 off... exact: -0.005)
    expect(out.splits[1].amount).toBeGreaterThanOrEqual(1400 - 1);
    expect(out.splits[1].amount).toBeLessThanOrEqual(1400);
  });

  it("names who received a leftover cent", () => {
    const r = receipt({ items: [item("A", 1000)], total: 1000 });
    const out = computeSplit(r, ["A", "B", "C"], plan(), false);
    expect(out.reasoning).toMatch(/1 leftover cent from rounding went to A/);
  });

  it("excluded items scale the rounding along with the surcharge", () => {
    const r = receipt({ items: [item("A", 1000), item("B", 3000)], tax: 400, total: 4396, rounding: -4 });
    const p = plan({ assignments: [{ itemIndex: 0, people: ["A", "B"] }], defaultRule: "exclude" });
    const out = computeSplit(r, ["A", "B"], p, true);
    // covers 1000 of 4000 = 25%: 25% of the 4.00 tax (1.00) and of the -0.04 rounding (-0.01)
    expect(sum(out.splits)).toBe(1000 + 100 - 1);
  });

  it("property: always reconciles to the printed total, for any people, tax payers, exclusions and rounding", () => {
    let seed = 99;
    const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
    for (let n = 0; n < 400; n++) {
      const people = Array.from({ length: 1 + Math.floor(rnd() * 5) }, (_, i) => `P${i}`);
      const items = Array.from({ length: 1 + Math.floor(rnd() * 6) }, (_, i) => item(`I${i}`, 1 + Math.floor(rnd() * 5000)));
      const subtotal = items.reduce((s, it) => s + it.totalPrice, 0);
      const tax = rnd() < 0.7 ? Math.floor(subtotal * rnd() * 0.16) : 0;
      const rounding = Math.floor(rnd() * 9) - 4; // -4..+4 sen
      const total = subtotal + tax + rounding;
      const assignments = items
        .map((_, i) => ({ itemIndex: i, people: people.filter(() => rnd() < 0.5) }))
        .filter((a) => a.people.length > 0 && rnd() < 0.6);
      const p = plan({
        assignments,
        defaultRule: rnd() < 0.8 ? "equal" : "exclude",
        taxPayers: rnd() < 0.3 ? [people[Math.floor(rnd() * people.length)]] : undefined,
      });
      const out = computeSplit(receipt({ items, tax, total, rounding }), people, p, true);
      const covered = items.reduce((s, it, i) => s + (assignments.some((a) => a.itemIndex === i) || p.defaultRule === "equal" ? it.totalPrice : 0), 0);
      const scale = (x: number) => (subtotal > 0 && covered > 0 ? Math.round((x * covered) / subtotal) : 0);
      expect(sum(out.splits)).toBe(covered + scale(Math.max(0, total - rounding - subtotal)) + scale(rounding));
      for (const s of out.splits) expect(s.amount).toBeGreaterThanOrEqual(0);
    }
  });
});
