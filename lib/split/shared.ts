import { billMe } from "@/lib/split/participants";
import { aggregateBalances, settleView, type PersonBalance } from "@/lib/split/settle";
import type { Cents, Participant, Settlement, SplitRecord } from "@/types/domain";

// A bill a friend shared with me, seen from MY side: only what runs between me and the friend who owns it.
// What they owe other people on the bill is theirs to track, so it never reaches my balances.

export interface SharedForBalances {
  id: string;
  title: string;
  createdAt: string | number | Date;
  currency: string;
  split: SplitRecord[] | null;
  settlement: Settlement | null;
  participants: Participant[] | null;
  /** What the owner calls me on that bill. */
  asName: string;
  owner: { id: string; name: string | null; username: string | null };
}

const norm = (s: string) => s.trim().toLowerCase();

export const ownerLabel = (o: { name: string | null; username: string | null }) => (o.name ?? "").trim() || (o.username ? `@${o.username}` : "A friend");

/**
 * Balances with the friends who shared bills with me: one entry per friend and currency, "owes you" when they
 * fronted nothing for me and I owe them, and so on. Only transfers between me (as the owner names me) and the
 * owner count, and only unpaid ones. Entries carry `shared`, so the screen opens the read-only bill.
 */
export function sharedBalances(bills: SharedForBalances[], today: string): PersonBalance[] {
  const merged = new Map<string, PersonBalance>();
  for (const b of bills) {
    if (!b.settlement || !b.split) continue;
    const ownerMe = billMe({ participants: b.participants ?? undefined, settlement: b.settlement });
    if (!ownerMe) continue; // hosted for others or an old bill: the owner isn't in it, nothing runs to them
    const asMe = aggregateBalances(
      [{ id: b.id, title: b.title, createdAt: b.createdAt, currency: b.currency, split: b.split, settlement: { ...b.settlement, me: b.asName } }],
      today,
    );
    for (const e of asMe) {
      if (norm(e.name) !== norm(ownerMe)) continue;
      const key = `${b.currency}|u:${b.owner.id}`;
      const cur = merged.get(key) ?? { key, name: ownerLabel(b.owner), currency: b.currency, net: 0, bills: [], shared: { ownerId: b.owner.id }, userId: b.owner.id };
      cur.net += e.net;
      cur.bills.push(...e.bills.map((x) => ({ ...x, shared: true as const })));
      merged.set(key, cur);
    }
  }
  return [...merged.values()].filter((p) => p.net !== 0);
}

/**
 * My own balances and the ones from friends' shared bills, as one list. A linked friend is a single row however the
 * bills name them (the key is the account), and a row with both kinds of bill is not "shared": its Mark all paid
 * and Nudge only ever touch the bills that are mine (each bill row says which it is).
 */
export function mergeBalances(own: PersonBalance[], fromFriends: PersonBalance[]): PersonBalance[] {
  const byKey = new Map<string, PersonBalance>(own.map((p) => [p.key, { ...p, bills: [...p.bills] }]));
  for (const f of fromFriends) {
    const cur = byKey.get(f.key);
    if (!cur) {
      byKey.set(f.key, f);
      continue;
    }
    cur.net += f.net;
    cur.bills.push(...f.bills);
  }
  return [...byKey.values()].filter((p) => p.net !== 0).sort((a, b) => Math.abs(b.net) - Math.abs(a.net) || a.name.localeCompare(b.name));
}

export interface MySharedSummary {
  /** What I still owe the owner on this bill, and what they still owe me (bills where I fronted something). */
  iOwe: Cents;
  theyOwe: Cents;
  /** Transfers between me and the owner that are still open: the ones I can say "I've paid" for. */
  open: { key: string; amount: Cents; direction: "iOwe" | "theyOwe" }[];
  settled: boolean;
}

/** One shared bill, from my side. */
export function mySharedSummary(b: Pick<SharedForBalances, "split" | "settlement" | "participants" | "asName">, today: string): MySharedSummary | null {
  if (!b.split || !b.settlement) return null;
  const ownerMe = billMe({ participants: b.participants ?? undefined, settlement: b.settlement });
  const view = settleView(b.split, b.settlement, today);
  if (!ownerMe || !view || !view.ok) return null;
  const me = norm(b.asName);
  const owner = norm(ownerMe);
  const mine = view.transfers.filter((t) => (norm(t.from) === me && norm(t.to) === owner) || (norm(t.to) === me && norm(t.from) === owner));
  const open = mine.filter((t) => !t.paid).map((t) => ({ key: t.key, amount: t.amount, direction: norm(t.from) === me ? ("iOwe" as const) : ("theyOwe" as const) }));
  return {
    iOwe: open.filter((o) => o.direction === "iOwe").reduce((a, o) => a + o.amount, 0),
    theyOwe: open.filter((o) => o.direction === "theyOwe").reduce((a, o) => a + o.amount, 0),
    open,
    settled: mine.length > 0 && open.length === 0,
  };
}
