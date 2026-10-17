import { describe, expect, it } from "vitest";
import { billSchema } from "./bill";

const base = {
  clientId: "6f9619ff-8b86-4011-b42d-00c04fc964ff",
  billTitle: "Dinner",
  totalAmount: 12360,
  currency: "RM",
  data: {
    split: [
      { name: "Alexis", amount: 6180, items: "x" },
      { name: "Riley", amount: 6180, items: "y" },
    ],
    items: { items: [{ name: "Food", quantity: 1, unitPrice: 12360, totalPrice: 12360 }], tax: 0, total: 12360, currency: "RM" },
    people: ["Alexis", "Riley"],
    reasoning: "",
  },
};
const withData = (extra: object) => ({ ...base, data: { ...base.data, ...extra } });

describe("billSchema: settlement", () => {
  it("accepts a bill saved before settle-up existed, unchanged", () => {
    const out = billSchema.parse(base);
    expect(out.data.settlement).toBeUndefined();
  });

  it("keeps the settlement through a parse (z.object would silently strip it otherwise)", () => {
    const settlement = {
      me: "Alexis",
      paidBy: [{ name: "Alexis", amount: 12360 }],
      paid: { "Riley->Alexis": { at: "2026-10-16", amount: 6180 } },
      dueDate: "2026-10-23",
    };
    expect(billSchema.parse(withData({ settlement })).data.settlement).toEqual(settlement);
  });

  it("keeps rounding and cash rounding too (the same trap, fixed earlier)", () => {
    const out = billSchema.parse(withData({
      cashRounding: { unit: 5, absorber: "Alexis" },
      items: { ...base.data.items, rounding: -2 },
    }));
    expect(out.data.cashRounding).toEqual({ unit: 5, absorber: "Alexis" });
    expect(out.data.items.rounding).toBe(-2);
  });

  it.each([
    ["a malformed due date", { dueDate: "next friday" }],
    ["a malformed paid date", { paid: { "A->B": { at: "yesterday", amount: 1 } } }],
    ["a fractional amount", { paidBy: [{ name: "Alexis", amount: 12.5 }] }],
    ["a negative amount", { paidBy: [{ name: "Alexis", amount: -1 }] }],
    ["too many payers", { paidBy: Array.from({ length: 51 }, (_, i) => ({ name: `P${i}`, amount: 1 })) }],
    ["too many paid marks", { paid: Object.fromEntries(Array.from({ length: 101 }, (_, i) => [`A${i}->B`, { at: "2026-10-16", amount: 1 }])) }],
  ])("rejects %s", (_label, patch) => {
    const settlement = { paidBy: [{ name: "Alexis", amount: 12360 }], paid: {}, ...patch };
    expect(billSchema.safeParse(withData({ settlement })).success).toBe(false);
  });
});

describe("billSchema: participants", () => {
  const me = { name: "Alexis", me: true as const };
  const settle = (m?: string) => ({ settlement: { ...(m ? { me: m } : {}), paidBy: [{ name: "Alexis", amount: 12360 }], paid: {} } });

  it("accepts a bill with the user marked, and keeps the participants", () => {
    const out = billSchema.parse(withData({ participants: [me, { name: "Riley" }], ...settle("Alexis") }));
    expect(out.data.participants).toEqual([me, { name: "Riley" }]);
  });
  it("accepts a bill split for others (nobody is me, settlement.me absent)", () => {
    expect(billSchema.safeParse(withData({ participants: [{ name: "Alexis" }, { name: "Riley" }], ...settle() })).success).toBe(true);
  });
  it("still accepts old bills: no participants, and any settlement.me", () => {
    expect(billSchema.safeParse(withData(settle("Whoever"))).success).toBe(true);
  });
  it.each([
    ["participants that differ from people", { participants: [me, { name: "Sam" }] }],
    ["a duplicate name ignoring case", { people: ["Alexis", "alexis"], split: [{ name: "Alexis", amount: 1, items: "x" }, { name: "alexis", amount: 1, items: "x" }], participants: [{ name: "Alexis" }, { name: "alexis" }] }],
    ["a name containing ->", { people: ["A->B", "Riley"], participants: [{ name: "A->B" }, { name: "Riley" }] }],
    ["two people marked me", { participants: [me, { name: "Riley", me: true as const }] }],
    ["a me that links another account", { participants: [{ ...me, userId: "u1" }, { name: "Riley" }] }],
    ["a settlement.me that disagrees", { participants: [me, { name: "Riley" }], ...settle("Riley") }],
    ["a settlement.me on a split for others", { participants: [{ name: "Alexis" }, { name: "Riley" }], ...settle("Alexis") }],
  ])("rejects %s", (_label, patch) => {
    expect(billSchema.safeParse(withData(patch)).success).toBe(false);
  });
});
