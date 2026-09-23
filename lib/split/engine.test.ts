import { describe, expect, it } from "vitest";
import { computeSplit } from "./engine";
import { SplitReconciliationError, type AssignmentPlan, type Receipt } from "@/types/domain";

function receipt(overrides: Partial<Receipt> = {}): Receipt {
  return {
    items: [],
    tax: 0,
    total: 0,
    currency: "RM",
    ...overrides,
  };
}

function plan(overrides: Partial<AssignmentPlan> = {}): AssignmentPlan {
  return {
    assignments: [],
    defaultRule: "equal",
    notes: "",
    ...overrides,
  };
}

function totalOf(splits: { amount: number }[]): number {
  return splits.reduce((a, s) => a + s.amount, 0);
}

describe("computeSplit", () => {
  it("splits an equal-share item with an indivisible remainder correctly (RM 10.00 / 3)", () => {
    const r = receipt({
      items: [{ name: "Nasi Lemak Set", quantity: 1, unitPrice: 1000, totalPrice: 1000 }],
      total: 1000,
    });
    const result = computeSplit(r, ["A", "B", "C"], plan(), false);
    expect(totalOf(result.splits)).toBe(1000);
    // No one should get 0, and no single person should be forced to
    // silently absorb the whole 1-cent remainder every time.
    const amounts = result.splits.map((s) => s.amount).sort((a, b) => a - b);
    expect(amounts).toEqual([333, 333, 334]);
  });

  it("gives a single-owner item entirely to the assigned person", () => {
    const r = receipt({
      items: [
        { name: "Teh Tarik", quantity: 1, unitPrice: 500, totalPrice: 500 },
        { name: "Nasi Lemak", quantity: 1, unitPrice: 1000, totalPrice: 1000 },
      ],
      total: 1500,
    });
    const p = plan({
      assignments: [{ itemIndex: 0, people: ["Pravin"] }],
      defaultRule: "equal",
    });
    const result = computeSplit(r, ["Pravin", "Aisha"], p, false);
    const pravin = result.splits.find((s) => s.name === "Pravin")!;
    const aisha = result.splits.find((s) => s.name === "Aisha")!;
    // Pravin: full drink (500) + half the rice (500) = 1000
    expect(pravin.amount).toBe(1000);
    // Aisha: just half the rice = 500
    expect(aisha.amount).toBe(500);
    expect(totalOf(result.splits)).toBe(1500);
  });

  it("respects explicit weights within an assignment", () => {
    const r = receipt({
      items: [{ name: "Pizza", quantity: 1, unitPrice: 1000, totalPrice: 1000 }],
      total: 1000,
    });
    const p = plan({
      assignments: [{ itemIndex: 0, people: ["A", "B"], weights: [3, 1] }],
    });
    const result = computeSplit(r, ["A", "B"], p, false);
    expect(result.splits.find((s) => s.name === "A")!.amount).toBe(750);
    expect(result.splits.find((s) => s.name === "B")!.amount).toBe(250);
  });

  it("prorates tax/service charge by food cost when applyTax is true", () => {
    const r = receipt({
      items: [
        { name: "A's Item", quantity: 1, unitPrice: 1000, totalPrice: 1000 },
        { name: "B's Item", quantity: 1, unitPrice: 2000, totalPrice: 2000 },
      ],
      total: 3300, // 300 surcharge (10% service charge) on a 3000 subtotal
    });
    const p = plan({
      assignments: [
        { itemIndex: 0, people: ["A"] },
        { itemIndex: 1, people: ["B"] },
      ],
    });
    const result = computeSplit(r, ["A", "B"], p, true);
    // A: 1000 + 10% of 1000 = 1100; B: 2000 + 10% of 2000 = 2200
    expect(result.splits.find((s) => s.name === "A")!.amount).toBe(1100);
    expect(result.splits.find((s) => s.name === "B")!.amount).toBe(2200);
    expect(totalOf(result.splits)).toBe(3300);
  });

  it("excludes tax/service charge entirely when applyTax is false", () => {
    const r = receipt({
      items: [{ name: "Item", quantity: 1, unitPrice: 1000, totalPrice: 1000 }],
      total: 1300, // would-be 300 surcharge, ignored
    });
    const result = computeSplit(r, ["A", "B"], plan(), false);
    expect(totalOf(result.splits)).toBe(1000);
  });

  it("handles a zero surcharge without a divide-by-zero or a phantom line", () => {
    const r = receipt({
      items: [{ name: "Item", quantity: 1, unitPrice: 1000, totalPrice: 1000 }],
      total: 1000,
    });
    const result = computeSplit(r, ["A", "B"], plan(), true);
    expect(totalOf(result.splits)).toBe(1000);
  });

  it("gives a person 0 when nothing is assigned to them and defaultRule is exclude", () => {
    const r = receipt({
      items: [{ name: "Item", quantity: 1, unitPrice: 1000, totalPrice: 1000 }],
      total: 1000,
    });
    const p = plan({
      assignments: [{ itemIndex: 0, people: ["A"] }],
      defaultRule: "exclude",
    });
    const result = computeSplit(r, ["A", "B"], p, false);
    expect(result.splits.find((s) => s.name === "A")!.amount).toBe(1000);
    expect(result.splits.find((s) => s.name === "B")!.amount).toBe(0);
    expect(totalOf(result.splits)).toBe(1000);
  });

  it("handles a negative (discount) line item", () => {
    const r = receipt({
      items: [
        { name: "Burger", quantity: 1, unitPrice: 1500, totalPrice: 1500 },
        { name: "Coupon Discount", quantity: 1, unitPrice: -300, totalPrice: -300 },
      ],
      total: 1200,
    });
    const result = computeSplit(r, ["A", "B"], plan(), false);
    expect(totalOf(result.splits)).toBe(1200);
  });

  it("derives the surcharge from total-subtotal, not from receipt.tax, for receipts with untracked service charges", () => {
    // Simulates a Malaysian receipt where `tax` only captures SST and the
    // service charge is a separate, unlabeled line baked into the total.
    const r = receipt({
      items: [{ name: "Item", quantity: 1, unitPrice: 1000, totalPrice: 1000 }],
      tax: 60, // SST only, as extracted
      total: 1160, // but the real total also includes a 100-cent service charge
    });
    const result = computeSplit(r, ["A"], plan(), true);
    // Must reconcile to the REAL total (1160), not to subtotal+tax (1060).
    expect(totalOf(result.splits)).toBe(1160);
  });

  it("handles a large receipt across many people without losing a cent", () => {
    const items = Array.from({ length: 40 }, (_, i) => ({
      name: `Item ${i}`,
      quantity: 1,
      unitPrice: 100 + i * 37,
      totalPrice: 100 + i * 37,
    }));
    const subtotal = items.reduce((s, it) => s + it.totalPrice, 0);
    const r = receipt({ items, total: Math.round(subtotal * 1.1) });
    const people = Array.from({ length: 8 }, (_, i) => `Person${i}`);
    // Assign every 5th item explicitly, let the rest fall to defaultRule.
    const assignments = items
      .map((_, i) => i)
      .filter((i) => i % 5 === 0)
      .map((i) => ({ itemIndex: i, people: [people[i % people.length]] }));
    const result = computeSplit(r, people, plan({ assignments }), true);
    expect(totalOf(result.splits)).toBe(r.total);
    expect(result.splits.length).toBe(8);
  });

  it("throws SplitReconciliationError with a diagnostic message if the invariant is ever violated", () => {
    // Not reachable through normal apportion()-mediated paths — this just
    // confirms the error class carries expected/actual for debugging if
    // it ever were to fire.
    const err = new SplitReconciliationError(1000, 999);
    expect(err.expected).toBe(1000);
    expect(err.actual).toBe(999);
    expect(err.message).toContain("1000");
    expect(err.message).toContain("999");
  });

  it("throws if the people list is empty", () => {
    const r = receipt({ items: [{ name: "X", quantity: 1, unitPrice: 100, totalPrice: 100 }], total: 100 });
    expect(() => computeSplit(r, [], plan(), false)).toThrow();
  });

  it("dedupes a duplicate itemIndex in the plan defensively (last one wins)", () => {
    const r = receipt({
      items: [{ name: "Item", quantity: 1, unitPrice: 1000, totalPrice: 1000 }],
      total: 1000,
    });
    const p = plan({
      assignments: [
        { itemIndex: 0, people: ["A"] },
        { itemIndex: 0, people: ["B"] }, // XGrammar can't prevent this; engine must not double-count
      ],
    });
    const result = computeSplit(r, ["A", "B"], p, false);
    expect(totalOf(result.splits)).toBe(1000);
  });

  it("dedupes a repeated name within one assignment's people array", () => {
    const r = receipt({
      items: [{ name: "Item", quantity: 1, unitPrice: 999, totalPrice: 999 }],
      total: 999,
    });
    const p = plan({
      assignments: [{ itemIndex: 0, people: ["A", "A", "B"] }], // uniqueItems not grammar-enforced
    });
    const result = computeSplit(r, ["A", "B"], p, false);
    // Should be split between A and B (2 shares), not A getting 2/3.
    expect(totalOf(result.splits)).toBe(999);
    const a = result.splits.find((s) => s.name === "A")!.amount;
    const b = result.splits.find((s) => s.name === "B")!.amount;
    expect(Math.abs(a - b)).toBeLessThanOrEqual(1);
  });

  it("ignores a name in the plan that isn't in the people list", () => {
    const r = receipt({
      items: [{ name: "Item", quantity: 1, unitPrice: 1000, totalPrice: 1000 }],
      total: 1000,
    });
    const p = plan({ assignments: [{ itemIndex: 0, people: ["Ghost"] }] });
    const result = computeSplit(r, ["A", "B"], p, false);
    // Ghost isn't a real person -> no explicit target survives -> falls
    // back to defaultRule "equal" across the real people list.
    expect(totalOf(result.splits)).toBe(1000);
    expect(result.splits.every((s) => s.amount > 0)).toBe(true);
  });

  it("holds the reconciliation invariant across many random receipts and plans", () => {
    let seed = 7;
    const rand = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff;
    };
    const randInt = (max: number) => Math.floor(rand() * max);

    for (let trial = 0; trial < 10000; trial++) {
      const itemCount = 1 + randInt(10);
      const items = Array.from({ length: itemCount }, (_, i) => {
        const price = 50 + randInt(5000);
        return { name: `Item${i}`, quantity: 1, unitPrice: price, totalPrice: price };
      });
      const subtotal = items.reduce((s, it) => s + it.totalPrice, 0);
      const surchargePct = randInt(20); // 0-19%
      const total = Math.round(subtotal * (1 + surchargePct / 100));

      const peopleCount = 1 + randInt(6);
      const people = Array.from({ length: peopleCount }, (_, i) => `P${i}`);

      // Randomly assign some items explicitly, leave others to defaultRule.
      const explicitlyAssigned = items.map((_, i) => i).filter(() => rand() < 0.5);
      const assignments = explicitlyAssigned.map((i) => {
        const n = 1 + randInt(peopleCount);
        const chosen = Array.from(new Set(Array.from({ length: n }, () => people[randInt(peopleCount)])));
        return { itemIndex: i, people: chosen };
      });

      const applyTax = rand() < 0.5;
      const defaultRule = rand() < 0.5 ? "equal" : "exclude";
      const p = plan({ assignments, defaultRule });

      const result = computeSplit(receipt({ items, total }), people, p, applyTax);

      // With defaultRule "equal", every item always resolves to a
      // non-empty target (explicit assignment, or the full people list) —
      // full coverage is guaranteed, so the split must reconcile to the
      // whole bill. With "exclude", any item that was never explicitly
      // assigned is legitimately dropped from the split (nobody in this
      // split pays for it) — the expected total scales down to match,
      // using the identical coverage-scaled surcharge formula the engine
      // itself uses, so this is still checking that apportion()'s calls
      // sum correctly, not re-deriving business logic.
      const coveredIndices =
        defaultRule === "equal" ? items.map((_, i) => i) : explicitlyAssigned;
      const coveredSubtotal = coveredIndices.reduce((s, i) => s + items[i].totalPrice, 0);
      const fullSurcharge = Math.max(0, total - subtotal);
      const surchargeForCovered =
        applyTax && subtotal > 0 ? Math.round((fullSurcharge * coveredSubtotal) / subtotal) : 0;
      const expected = coveredSubtotal + surchargeForCovered;

      expect(totalOf(result.splits)).toBe(expected);
      // Never collect more than the receipt's own total.
      expect(totalOf(result.splits)).toBeLessThanOrEqual(applyTax ? total : subtotal);
    }
  });
});
