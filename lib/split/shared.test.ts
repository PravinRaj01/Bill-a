import { describe, expect, it } from "vitest";
import { aggregateBalances } from "./settle";

import { mergeBalances, mySharedSummary, sharedBalances, type SharedForBalances } from "./shared";

const owner = { id: "owner-1", name: "Alexis", username: "alexis" };
const bill = (over: Partial<SharedForBalances> = {}): SharedForBalances => ({
  id: "b1",
  title: "Dinner",
  createdAt: "2026-01-01",
  currency: "RM",
  split: [
    { name: "Alexis", amount: 3000, items: "x" },
    { name: "Riley", amount: 3000, items: "x" },
    { name: "Casey", amount: 3000, items: "x" },
  ],
  settlement: { me: "Alexis", paidBy: [{ name: "Alexis", amount: 9000 }], paid: {} },
  participants: null,
  asName: "Riley",
  owner,
  ...over,
});

describe("sharedBalances", () => {
  it("counts only what runs between me and the owner (not what Casey owes them)", () => {
    const [p] = sharedBalances([bill()], "2026-01-02");
    expect(p).toMatchObject({ name: "Alexis", currency: "RM", net: -3000, shared: { ownerId: "owner-1" } });
    expect(p.bills).toHaveLength(1);
  });

  it("drops paid transfers and merges several bills from the same friend", () => {
    const paid = bill({ id: "b2", settlement: { me: "Alexis", paidBy: [{ name: "Alexis", amount: 9000 }], paid: { "Riley->Alexis": { at: "2026-01-02", amount: 3000 } } } });
    const [p] = sharedBalances([bill(), paid], "2026-01-03");
    expect(p.net).toBe(-3000); // only the unpaid one counts
    expect(p.bills.map((b) => b.id)).toEqual(["b1"]);
    const both = sharedBalances([bill(), bill({ id: "b3" })], "2026-01-03");
    expect(both).toHaveLength(1);
    expect(both[0].net).toBe(-6000);
  });

  it("shows what the owner owes me when I fronted the money", () => {
    const [p] = sharedBalances([bill({ settlement: { me: "Alexis", paidBy: [{ name: "Riley", amount: 9000 }], paid: {} } })], "2026-01-02");
    expect(p.net).toBe(3000);
  });

  it("ignores a bill split for others (the owner isn't in it), one with no settlement, and different currencies stay apart", () => {
    const hosted = bill({ participants: [{ name: "Riley" }, { name: "Casey" }], settlement: { paidBy: [{ name: "Casey", amount: 9000 }], paid: {} } });
    expect(sharedBalances([hosted, bill({ settlement: null })], "2026-01-02")).toEqual([]);
    expect(sharedBalances([bill(), bill({ id: "b4", currency: "$" })], "2026-01-02")).toHaveLength(2);
  });

  it("names a friend with no display name by @username", () => {
    const [p] = sharedBalances([bill({ owner: { id: "o2", name: null, username: "alexis" } })], "2026-01-02");
    expect(p.name).toBe("@alexis");
  });
});

describe("mySharedSummary", () => {
  it("lists my open transfers with the owner, and says settled once they're paid", () => {
    expect(mySharedSummary(bill(), "2026-01-02")).toEqual({ iOwe: 3000, theyOwe: 0, open: [{ key: "Riley->Alexis", amount: 3000, direction: "iOwe" }], settled: false });
    const paid = bill({ settlement: { me: "Alexis", paidBy: [{ name: "Alexis", amount: 9000 }], paid: { "Riley->Alexis": { at: "2026-01-02", amount: 3000 } } } });
    expect(mySharedSummary(paid, "2026-01-03")).toMatchObject({ iOwe: 0, open: [], settled: true });
  });
  it("is null without a settlement", () => {
    expect(mySharedSummary(bill({ settlement: null }), "2026-01-02")).toBeNull();
  });
});

describe("mergeBalances", () => {
  it("shows a linked friend as one row, however their own bills named them", () => {
    const own = aggregateBalances(
      [
        {
          id: "own1",
          title: "Lunch",
          createdAt: "2026-01-01",
          currency: "RM",
          split: [
            { name: "Me", amount: 2000 },
            { name: "Pravin", amount: 2000 },
          ],
          settlement: { me: "Me", paidBy: [{ name: "Me", amount: 4000 }], paid: {} },
          linked: { pravin: "owner-1" },
        },
      ],
      "2026-01-02",
    );
    expect(own[0].key).toBe("RM|u:owner-1");
    const theirs = sharedBalances([bill()], "2026-01-02"); // Alexis (owner-1) is owed 30.00 by me
    const merged = mergeBalances(own, theirs);
    expect(merged).toHaveLength(1);
    expect(merged[0].net).toBe(2000 - 3000);
    expect(merged[0].name).toBe("Pravin"); // my own word for them
    expect(merged[0].shared).toBeUndefined(); // mixed: Mark all paid only touches my bills
    expect(merged[0].bills.map((b) => !!b.shared)).toEqual([false, true]);
  });

  it("a row of friends' bills only stays shared", () => {
    const merged = mergeBalances([], sharedBalances([bill()], "2026-01-02"));
    expect(merged[0].shared).toEqual({ ownerId: "owner-1" });
  });
});
