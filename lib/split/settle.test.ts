import { describe, expect, it } from "vitest";
import type { Settlement } from "@/types/domain";
import {
  addDays,
  defaultSettlement,
  normalizeSettlement,
  aggregateBalances,
  balanceNudgeText,
  balanceTotals,
  balances,
  daysUntil,
  groupNudgeText,
  guessMe,
  localDate,
  markAllPaid,
  markPaid,
  minimalTransfers,
  nudgeText,
  paidMatches,
  settleView,
  shortDate,
  singlePayer,
  transferKey,
  unmarkPaid,
  withPayers,
  type Share,
} from "./settle";

const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);
const shares3: Share[] = [
  { name: "Alexis", amount: 4120 },
  { name: "Riley", amount: 4120 },
  { name: "Aisha", amount: 4120 },
];
const settle = (o: Partial<Settlement> = {}): Settlement => ({ paidBy: singlePayer("Alexis", 12360), paid: {}, me: "Alexis", ...o });
const view = (shares: Share[], s: Settlement | undefined, today = "2026-10-16") => {
  const v = settleView(shares, s, today);
  if (!v || !v.ok) throw new Error(`expected an ok view, got ${JSON.stringify(v)}`);
  return v;
};

describe("balances / paidMatches", () => {
  it("fronted minus owed, summing to zero", () => {
    const b = balances(shares3, singlePayer("Alexis", 12360));
    expect(b).toEqual([{ name: "Alexis", net: 8240 }, { name: "Riley", net: -4120 }, { name: "Aisha", net: -4120 }]);
    expect(sum(b.map((x) => x.net))).toBe(0);
  });

  it("a payer who isn't among the sharers (they treated everyone) is still counted", () => {
    const b = balances(shares3.slice(1), singlePayer("Boss", 8240));
    expect(b.find((x) => x.name === "Boss")?.net).toBe(8240);
    expect(sum(b.map((x) => x.net))).toBe(0);
  });

  it("detects fronted money that doesn't match what is owed", () => {
    expect(paidMatches(shares3, singlePayer("Alexis", 12360))).toBe(true);
    expect(paidMatches(shares3, singlePayer("Alexis", 12000))).toBe(false);
  });
});

describe("minimalTransfers", () => {
  it("single payer: everyone owes the payer their share", () => {
    const t = minimalTransfers(balances(shares3, singlePayer("Alexis", 12360)));
    expect(t.map((x) => [x.from, x.to, x.amount])).toEqual([["Riley", "Alexis", 4120], ["Aisha", "Alexis", 4120]]);
  });

  it("several payers: the fewest simple hand-overs, and Alexis ends up square", () => {
    // owed 41.20 each. Alexis fronted 80.00, Riley 43.60 -> Alexis +38.80, Riley +2.40, Aisha -41.20
    const t = minimalTransfers(balances(shares3, [{ name: "Alexis", amount: 8000 }, { name: "Riley", amount: 4360 }]));
    expect(t.map((x) => [x.from, x.to, x.amount])).toEqual([["Aisha", "Alexis", 3880], ["Aisha", "Riley", 240]]);
  });

  it("nothing to do when everyone paid their own share, or when the books don't balance", () => {
    expect(minimalTransfers(balances(shares3, shares3.map((s) => ({ name: s.name, amount: s.amount }))))).toEqual([]);
    expect(minimalTransfers([{ name: "A", net: 100 }, { name: "B", net: -50 }])).toEqual([]);
  });

  it("a key per pair", () => {
    expect(transferKey("Riley", "Alexis")).toBe("Riley->Alexis");
  });

  it("property: transfers always settle every balance exactly, in at most n-1 steps", () => {
    let seed = 4242;
    const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
    for (let n = 0; n < 500; n++) {
      const people = 2 + Math.floor(rnd() * 7);
      const shares: Share[] = Array.from({ length: people }, (_, i) => ({ name: `P${i}`, amount: Math.floor(rnd() * 20000) }));
      const total = sum(shares.map((s) => s.amount));
      // split the fronted money randomly among a random subset of payers, to the cent
      const payers = shares.filter(() => rnd() < 0.5);
      const chosen = payers.length ? payers : [shares[0]];
      let left = total;
      const paidBy = chosen.map((p, i) => {
        const amount = i === chosen.length - 1 ? left : Math.floor(rnd() * (left + 1));
        left -= amount;
        return { name: p.name, amount };
      });
      const bal = balances(shares, paidBy);
      const transfers = minimalTransfers(bal);
      expect(transfers.length).toBeLessThanOrEqual(Math.max(0, bal.filter((b) => b.net !== 0).length - 1));
      // apply the transfers: everyone must end at exactly zero
      const end = new Map(bal.map((b) => [b.name, b.net]));
      for (const t of transfers) {
        expect(t.amount).toBeGreaterThan(0);
        end.set(t.from, end.get(t.from)! + t.amount);
        end.set(t.to, end.get(t.to)! - t.amount);
      }
      for (const v of end.values()) expect(v).toBe(0);
    }
  });
});

describe("settleView", () => {
  it("is null with no settlement", () => {
    expect(settleView(shares3, undefined, "2026-10-16")).toBeNull();
  });

  it("reports a mismatch instead of guessing", () => {
    const v = settleView(shares3, settle({ paidBy: singlePayer("Alexis", 10000) }), "2026-10-16");
    expect(v).toEqual({ ok: false, reason: "paid-mismatch", fronted: 10000, owed: 12360 });
  });

  it("open: nothing paid yet", () => {
    const v = view(shares3, settle());
    expect([v.owed, v.collected, v.outstanding, v.paidCount, v.status, v.dueIn]).toEqual([8240, 0, 8240, 0, "open", null]);
  });

  it("partly paid, then settled", () => {
    const s1 = markPaid(settle(), { key: "Riley->Alexis", amount: 4120 }, "2026-10-16");
    const v1 = view(shares3, s1);
    expect([v1.collected, v1.outstanding, v1.paidCount, v1.status]).toEqual([4120, 4120, 1, "open"]);
    expect(v1.transfers.find((t) => t.from === "Riley")?.paid).toBe(true);

    const s2 = markPaid(s1, { key: "Aisha->Alexis", amount: 4120 }, "2026-10-17");
    const v2 = view(shares3, s2);
    expect([v2.outstanding, v2.status]).toEqual([0, "settled"]);
  });

  it("a single person (nobody to collect from) is settled", () => {
    const v = view([{ name: "Alexis", amount: 5000 }], settle({ paidBy: singlePayer("Alexis", 5000) }));
    expect([v.transfers.length, v.status]).toEqual([0, "settled"]);
  });

  it("a paid mark stops counting when the amount owed changes, and is dropped from the saved copy", () => {
    const marked = markPaid(settle(), { key: "Riley->Alexis", amount: 4120 }, "2026-10-16");
    // the shares are adjusted: Riley now owes 50.00 and Aisha 32.40
    const shifted: Share[] = [{ name: "Alexis", amount: 4120 }, { name: "Riley", amount: 5000 }, { name: "Aisha", amount: 3240 }];
    const v = view(shifted, marked);
    expect(v.transfers.find((t) => t.from === "Riley")).toMatchObject({ amount: 5000, paid: false });
    expect(v.settlement.paid).toEqual({});
  });

  it("a mark for a transfer that no longer exists is dropped", () => {
    const s = settle({ paid: { "Ghost->Alexis": { at: "2026-10-01", amount: 100 } } });
    expect(view(shares3, s).settlement.paid).toEqual({});
  });

  describe("due dates", () => {
    const due = (dueDate: string, today = "2026-10-16") => view(shares3, settle({ dueDate }), today);
    it("open before, due on the day, overdue after", () => {
      expect([due("2026-10-19").status, due("2026-10-19").dueIn]).toEqual(["open", 3]);
      expect([due("2026-10-16").status, due("2026-10-16").dueIn]).toEqual(["due", 0]);
      expect([due("2026-10-14").status, due("2026-10-14").dueIn]).toEqual(["overdue", -2]);
    });
    it("settled wins over overdue", () => {
      const all = markAllPaid(settle({ dueDate: "2026-10-01" }), view(shares3, settle()).transfers, "2026-10-16");
      expect(view(shares3, all).status).toBe("settled");
    });
  });
});

describe("edits", () => {
  it("mark, unmark, mark all", () => {
    const t = view(shares3, settle()).transfers;
    const one = markPaid(settle(), t[0], "2026-10-16");
    expect(one.paid[t[0].key]).toEqual({ at: "2026-10-16", amount: t[0].amount });
    expect(unmarkPaid(one, t[0].key).paid).toEqual({});
    const all = markAllPaid(settle(), t, "2026-10-16");
    expect(Object.keys(all.paid).sort()).toEqual(t.map((x) => x.key).sort());
    // already-paid transfers keep their original date
    const again = markAllPaid(one, view(shares3, one).transfers, "2026-10-20");
    expect(again.paid[t[0].key].at).toBe("2026-10-16");
  });

  it("edits don't mutate their input", () => {
    const s = settle();
    markPaid(s, { key: "a->b", amount: 1 }, "2026-10-16");
    unmarkPaid(s, "a->b");
    expect(s.paid).toEqual({});
  });

  it("changing who paid clears the old marks but keeps me and the due date", () => {
    const s = withPayers(markPaid(settle({ dueDate: "2026-10-20" }), { key: "Riley->Alexis", amount: 4120 }, "2026-10-16"), singlePayer("Riley", 12360));
    expect(s).toMatchObject({ me: "Alexis", dueDate: "2026-10-20", paid: {}, paidBy: [{ name: "Riley", amount: 12360 }] });
  });
});

describe("guessMe", () => {
  it("matches the account's first name, ignoring case", () => {
    expect(guessMe(["Alexis Raj", "Riley"], "alexis raj")).toBe("Alexis Raj");
    expect(guessMe(["Alexis", "Riley"], "ALEXIS")).toBe("Alexis");
  });
  it("doesn't guess when it is ambiguous or unknown", () => {
    expect(guessMe(["Wei Jie", "Wei Ling"], "Wei Jie")).toBeUndefined();
    expect(guessMe(["Alexis", "Riley"], "Someone")).toBeUndefined();
    expect(guessMe(["Alexis"], null)).toBeUndefined();
  });
});

describe("dates", () => {
  it("daysUntil / addDays / localDate", () => {
    expect(daysUntil("2026-10-19", "2026-10-16")).toBe(3);
    expect(daysUntil("2026-03-30", "2026-03-28")).toBe(2); // across a DST change somewhere
    expect(addDays("2026-10-30", 3)).toBe("2026-11-02");
    expect(addDays("2026-12-31", 1)).toBe("2027-01-01");
    expect(localDate(new Date(2026, 9, 5))).toBe("2026-10-05");
    expect(shortDate("2026-10-16")).toBe("16 Oct");
  });
});

describe("aggregateBalances", () => {
  const dinner = {
    id: "b1", title: "Dinner", createdAt: "2026-10-10T12:00:00Z", currency: "RM", split: shares3, settlement: settle(),
  };
  const taxi = {
    id: "b2", title: "Taxi", createdAt: "2026-10-12T12:00:00Z", currency: "RM",
    split: [{ name: "alexis", amount: 1500 }, { name: "RILEY", amount: 1500 }],
    settlement: settle({ me: "alexis", paidBy: singlePayer("RILEY", 3000) }), // Riley paid; I owe him
  };

  it("nets what each person owes you across bills, matching names ignoring case", () => {
    const list = aggregateBalances([dinner, taxi], "2026-10-16");
    const riley = list.find((p) => p.name.toLowerCase() === "riley")!;
    expect(riley.net).toBe(4120 - 1500); // owes me 41.20 for dinner, I owe him 15.00 for the taxi
    expect(riley.bills.map((b) => [b.id, b.amount])).toEqual([["b1", 4120], ["b2", -1500]]);
    expect(riley.name).toBe("RILEY"); // the most recent bill's spelling
    const aisha = list.find((p) => p.name === "Aisha")!;
    expect(aisha.net).toBe(4120);
  });

  it("paid transfers don't count, and fully cancelling people drop out", () => {
    const paid = { ...dinner, settlement: markPaid(settle(), { key: "Riley->Alexis", amount: 4120 }, "2026-10-11") };
    const list = aggregateBalances([paid, taxi], "2026-10-16");
    expect(list.find((p) => p.name.toLowerCase() === "riley")!.net).toBe(-1500);
    const square = aggregateBalances([{ ...dinner, split: [{ name: "Alexis", amount: 1500 }, { name: "Riley", amount: 1500 }], settlement: settle({ paidBy: singlePayer("Alexis", 3000) }) }, taxi], "2026-10-16");
    expect(square.find((p) => p.name.toLowerCase() === "riley")).toBeUndefined();
  });

  it("ignores bills with no settlement, no 'me', a mismatch, or a debt between two other people", () => {
    const noSettlement = { ...dinner, id: "x1", settlement: undefined };
    const noMe = { ...dinner, id: "x2", settlement: settle({ me: undefined }) };
    const mismatch = { ...dinner, id: "x3", settlement: settle({ paidBy: singlePayer("Alexis", 1) }) };
    const others = { ...dinner, id: "x4", settlement: settle({ me: "Aisha", paidBy: singlePayer("Riley", 12360) }) };
    const list = aggregateBalances([noSettlement, noMe, mismatch, others], "2026-10-16");
    // only x4 counts: Aisha (me) owes Riley 41.20; Alexis owing Riley is not my business
    expect(list.map((p) => [p.name, p.net])).toEqual([["Riley", -4120]]);
  });

  it("keeps currencies apart and sorts the biggest balance first", () => {
    const usd = { ...dinner, id: "u1", currency: "$", split: [{ name: "Alexis", amount: 1000 }, { name: "Riley", amount: 1000 }], settlement: settle({ paidBy: singlePayer("Alexis", 2000) }) };
    const list = aggregateBalances([dinner, usd], "2026-10-16");
    expect(list.filter((p) => p.name === "Riley").map((p) => [p.currency, p.net])).toEqual([["RM", 4120], ["$", 1000]]);
    expect(Math.abs(list[0].net)).toBeGreaterThanOrEqual(Math.abs(list[list.length - 1].net));
  });

  it("totals", () => {
    const list = aggregateBalances([dinner, taxi], "2026-10-16");
    expect(balanceTotals(list)).toEqual({ owedToYou: 4120 + 2620, youOwe: 0 });
    expect(balanceTotals(aggregateBalances([taxi], "2026-10-16"))).toEqual({ owedToYou: 0, youOwe: 1500 });
  });
});

describe("messages", () => {
  it("one person's nudge names the person, bill, date, amount and items", () => {
    const text = nudgeText({ debtor: "Riley", payTo: "me", title: "Dinner", date: "2026-10-16", amount: 4120, currency: "RM", items: "Nasi lemak (x1/2), Teh tarik" });
    expect(text).toBe("Hi Riley! Your share for *Dinner* (16 Oct) is *RM41.20* (Nasi lemak (x1/2), Teh tarik). Please pay me when you can. Thanks! — via Bill.a");
  });

  it("leaves out share descriptions that are not item names", () => {
    for (const items of ["33.3% of the bill", "70% of the bill", "equal share of the remaining RM50.00", "amount you gave", "2 of 3 parts"]) {
      expect(nudgeText({ debtor: "A", payTo: "B", title: "T", date: "2026-10-16", amount: 100, currency: "RM", items })).toBe(
        "Hi A! Your share for *T* (16 Oct) is *RM1.00*. Please pay B when you can. Thanks! — via Bill.a",
      );
    }
  });

  it("leaves out empty items and clips long ones", () => {
    expect(nudgeText({ debtor: "A", payTo: "B", title: "T", date: "2026-10-16", amount: 100, currency: "RM", items: "(nothing assigned)" })).not.toMatch(/\(\(/);
    const long = nudgeText({ debtor: "A", payTo: "B", title: "T", date: "2026-10-16", amount: 100, currency: "RM", items: "x".repeat(300) });
    expect(long.length).toBeLessThan(260);
    expect(long).toMatch(/…/);
  });

  it("group and cross-bill messages list each line with the total", () => {
    expect(groupNudgeText({ title: "Dinner", date: "2026-10-16", payTo: "Alexis", currency: "RM", lines: [{ debtor: "Riley", amount: 4120 }, { debtor: "Aisha", amount: 4120 }] }))
      .toBe("*Dinner* (16 Oct) — still to pay Alexis:\n• Riley: RM41.20\n• Aisha: RM41.20\nThanks! — via Bill.a");
    expect(balanceNudgeText({ name: "Riley", payTo: "me", currency: "RM", lines: [{ title: "Dinner", date: "2026-10-10", amount: 4120 }, { title: "Taxi", date: "2026-10-12", amount: 1500 }] }))
      .toBe("Hi Riley! A reminder of what's still open:\n• Dinner (10 Oct): RM41.20\n• Taxi (12 Oct): RM15.00\nTotal *RM56.20* to me. Thanks! — via Bill.a");
  });
});

describe("defaultSettlement / normalizeSettlement", () => {
  it("the user fronted it if we know who that is, otherwise the first person", () => {
    expect(defaultSettlement(shares3, "Riley")).toEqual({ me: "Riley", paidBy: [{ name: "Riley", amount: 12360 }], paid: {} });
    expect(defaultSettlement(shares3)).toEqual({ paidBy: [{ name: "Alexis", amount: 12360 }], paid: {} });
    expect(defaultSettlement(shares3, "Ghost").paidBy[0].name).toBe("Alexis"); // not in the group
  });

  it("a single payer follows the bill when the shares change total", () => {
    const grown = shares3.map((s) => ({ ...s, amount: s.amount + 100 }));
    const n = normalizeSettlement(settle(), grown);
    expect(n.paidBy).toEqual([{ name: "Alexis", amount: 12660 }]);
    expect(settleView(grown, n, "2026-10-16")?.ok).toBe(true);
  });

  it("if the payer or 'me' left the group, the first person takes over and 'me' is forgotten", () => {
    const n = normalizeSettlement(settle({ paidBy: singlePayer("Alexis", 12360), me: "Alexis" }), shares3.slice(1));
    expect(n.paidBy[0].name).toBe("Riley");
    expect(n.me).toBeUndefined();
  });

  it("several payers are left as entered, so a mismatch shows instead of being hidden", () => {
    const multi = settle({ paidBy: [{ name: "Alexis", amount: 8000 }, { name: "Riley", amount: 4360 }] });
    const grown = shares3.map((s) => ({ ...s, amount: s.amount + 100 }));
    const n = normalizeSettlement(multi, grown);
    expect(n.paidBy).toEqual(multi.paidBy);
    expect(settleView(grown, n, "2026-10-16")).toMatchObject({ ok: false, reason: "paid-mismatch" });
  });

  it("is a no-op on an empty group", () => {
    const s = settle();
    expect(normalizeSettlement(s, [])).toBe(s);
  });
});
