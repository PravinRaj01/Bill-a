import { formatMoney } from "@/lib/money";
import { apportion } from "./apportion";
import type { Cents, Settlement } from "@/types/domain";

// Settle-up: who fronted a bill, who owes whom, who has paid back. Everything here is integer cents
// and deterministic, and nothing here touches a model or the network.
//
// The facts a person enters are stored (who paid, which transfers are marked paid, a due date).
// What is owed is always DERIVED from the shares people actually pay, so it can never disagree with
// them: change the shares and the transfers follow, and a "paid" mark that no longer matches the
// amount owed simply stops counting.

export interface Share {
  name: string;
  amount: Cents;
}

export interface Transfer {
  from: string;
  to: string;
  amount: Cents;
  /** "From->To": the key a paid mark is stored under. */
  key: string;
  paid: boolean;
}

export type SettleStatus = "settled" | "open" | "due" | "overdue";

export const transferKey = (from: string, to: string) => `${from}->${to}`;

const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);
const norm = (s: string) => s.trim().toLowerCase();

// ---- dates (local calendar dates as "YYYY-MM-DD", compared as UTC days so DST can't skew them) ----

const pad = (n: number) => String(n).padStart(2, "0");
export const localDate = (d: Date = new Date()): string => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;

function dayNumber(iso: string): number {
  const [y, m, d] = iso.split("-").map(Number);
  return Math.round(Date.UTC(y, m - 1, d) / 86_400_000);
}

/** Whole days from `today` to `due` (negative = overdue). */
export const daysUntil = (due: string, today: string): number => dayNumber(due) - dayNumber(today);

/** `today` + n days. */
export function addDays(today: string, n: number): string {
  const [y, m, d] = today.split("-").map(Number);
  return localDate(new Date(y, m - 1, d + n));
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
/** "16 Oct". English month names on purpose: the message text is English whatever the device locale. */
export function shortDate(d: string | number | Date): string {
  const date = typeof d === "string" && /^\d{4}-\d{2}-\d{2}$/.test(d) ? new Date(`${d}T00:00:00`) : new Date(d);
  return `${date.getDate()} ${MONTHS[date.getMonth()]}`;
}

// ---- who paid -------------------------------------------------------------------------------

export const singlePayer = (name: string, total: Cents): Settlement["paidBy"] => [{ name, amount: total }];

/** True when what was fronted adds up to what is owed (the one rule that makes the rest well-defined). */
export const paidMatches = (shares: Share[], paidBy: Settlement["paidBy"]): boolean =>
  sum(shares.map((s) => s.amount)) === sum(paidBy.map((p) => p.amount));

/**
 * Each person's position once the bill is paid: what they fronted minus what they owe.
 * Positive = they are owed money, negative = they owe. Sums to zero when `paidMatches`.
 */
export function balances(shares: Share[], paidBy: Settlement["paidBy"]): { name: string; net: Cents }[] {
  const fronted = new Map<string, number>();
  for (const p of paidBy) fronted.set(p.name, (fronted.get(p.name) ?? 0) + p.amount);
  const owes = new Map(shares.map((s) => [s.name, s.amount]));
  const names = [...shares.map((s) => s.name), ...[...fronted.keys()].filter((n) => !owes.has(n))];
  return names.map((name) => ({ name, net: (fronted.get(name) ?? 0) - (owes.get(name) ?? 0) }));
}

/**
 * Pairs debtors with creditors, biggest first, until everyone is square. Every step clears at least
 * one person, so there are at most (people - 1) transfers; with one payer it is simply "everyone owes
 * the payer their share". Ties break by the order people were listed, so the result never shuffles.
 */
export function minimalTransfers(bal: { name: string; net: Cents }[]): Omit<Transfer, "paid">[] {
  if (sum(bal.map((b) => b.net)) !== 0) return [];
  const order = (a: { amt: number; i: number }, b: { amt: number; i: number }) => b.amt - a.amt || a.i - b.i;
  const credit = bal.map((b, i) => ({ name: b.name, amt: b.net, i })).filter((b) => b.amt > 0).sort(order);
  const debt = bal.map((b, i) => ({ name: b.name, amt: -b.net, i })).filter((b) => b.amt > 0).sort(order);

  const out: Omit<Transfer, "paid">[] = [];
  let c = 0;
  let d = 0;
  while (c < credit.length && d < debt.length) {
    const amount = Math.min(credit[c].amt, debt[d].amt);
    out.push({ from: debt[d].name, to: credit[c].name, amount, key: transferKey(debt[d].name, credit[c].name) });
    credit[c].amt -= amount;
    debt[d].amt -= amount;
    if (credit[c].amt === 0) c++;
    if (debt[d].amt === 0) d++;
  }
  return out;
}

// ---- the bill's settle-up state -------------------------------------------------------------

export type SettleView =
  | { ok: false; reason: "paid-mismatch"; fronted: Cents; owed: Cents }
  | {
      ok: true;
      transfers: Transfer[];
      /** Total that has to change hands (the payer's own share never does). */
      owed: Cents;
      collected: Cents;
      outstanding: Cents;
      paidCount: number;
      status: SettleStatus;
      /** Days until the due date (negative = overdue); null with no due date. */
      dueIn: number | null;
      /** The settlement with stale paid marks dropped: save this, not the one you passed in. */
      settlement: Settlement;
    };

/** Settle-up state of one bill. `null` when it has no settlement. */
export function settleView(shares: Share[], settlement: Settlement | undefined, today: string): SettleView | null {
  if (!settlement || settlement.paidBy.length === 0) return null;
  if (!paidMatches(shares, settlement.paidBy)) {
    return { ok: false, reason: "paid-mismatch", fronted: sum(settlement.paidBy.map((p) => p.amount)), owed: sum(shares.map((s) => s.amount)) };
  }

  // A mark only counts if it is for the same pair AND the same amount as is owed now.
  const live: Settlement["paid"] = {};
  const transfers: Transfer[] = minimalTransfers(balances(shares, settlement.paidBy)).map((t) => {
    const mark = settlement.paid[t.key];
    const paid = !!mark && mark.amount === t.amount;
    if (paid) live[t.key] = mark;
    return { ...t, paid };
  });

  const owed = sum(transfers.map((t) => t.amount));
  const collected = sum(transfers.filter((t) => t.paid).map((t) => t.amount));
  const outstanding = owed - collected;
  const { status, dueIn } = settleStatus(outstanding, settlement.dueDate, today);

  return {
    ok: true,
    transfers,
    owed,
    collected,
    outstanding,
    paidCount: transfers.filter((t) => t.paid).length,
    status,
    dueIn,
    settlement: { ...settlement, paid: live },
  };
}

// ---- edits (immutable) ----------------------------------------------------------------------

export function markPaid(s: Settlement, t: Pick<Transfer, "key" | "amount">, today: string): Settlement {
  return { ...s, paid: { ...s.paid, [t.key]: { at: today, amount: t.amount } } };
}

export function unmarkPaid(s: Settlement, key: string): Settlement {
  const { [key]: _removed, ...rest } = s.paid;
  void _removed;
  return { ...s, paid: rest };
}

/** Mark every still-open transfer paid (ones already paid keep their original date). */
export function markAllPaid(s: Settlement, transfers: Transfer[], today: string): Settlement {
  return transfers.reduce((acc, t) => (t.paid ? acc : markPaid(acc, t, today)), s);
}

/** A new payer arrangement invalidates the old marks (the pairs change), so they are cleared. */
export function withPayers(s: Settlement | undefined, paidBy: Settlement["paidBy"]): Settlement {
  return { ...s, paidBy, paid: {} };
}

/** Which person is the signed-in user? The one whose first name matches the account's, else nobody. */
export function guessMe(people: string[], accountName?: string | null): string | undefined {
  const first = norm((accountName ?? "").split(/\s+/)[0] ?? "");
  if (!first) return undefined;
  const hits = people.filter((p) => norm(p.split(/\s+/)[0] ?? "") === first);
  return hits.length === 1 ? hits[0] : undefined; // two people with the same first name: don't guess
}

// ---- across bills ---------------------------------------------------------------------------

export interface BillForBalances {
  id: string;
  title: string;
  createdAt: string | number | Date;
  currency: string;
  split: Share[];
  settlement?: Settlement;
}

export interface PersonBalance {
  /** Stable id: currency + the lower-cased name. */
  key: string;
  name: string;
  currency: string;
  /** Positive: they owe you. Negative: you owe them. */
  net: Cents;
  bills: { id: string; title: string; createdAt: string | number | Date; amount: Cents; dueDate?: string }[];
}

/**
 * What each person owes you (or you them) across all bills, counting only transfers that involve
 * YOU on that bill (`settlement.me`) and are still unpaid. Two people are the same person if their
 * names match ignoring case and spaces; the display name is the most recent bill's spelling.
 * Amounts in different currencies are kept apart, never netted.
 */
export function aggregateBalances(bills: BillForBalances[], today: string): PersonBalance[] {
  const byPerson = new Map<string, PersonBalance>();
  const oldestFirst = [...bills].sort((a, b) => +new Date(a.createdAt) - +new Date(b.createdAt));

  for (const bill of oldestFirst) {
    const me = bill.settlement?.me;
    if (!me) continue;
    const view = settleView(bill.split, bill.settlement, today);
    if (!view || !view.ok) continue;

    for (const t of view.transfers) {
      if (t.paid) continue;
      const iAmCreditor = norm(t.to) === norm(me);
      const iAmDebtor = norm(t.from) === norm(me);
      if (!iAmCreditor && !iAmDebtor) continue; // a debt between two other people
      const other = iAmCreditor ? t.from : t.to;
      const signed = iAmCreditor ? t.amount : -t.amount;
      const key = `${bill.currency}|${norm(other)}`;
      const entry = byPerson.get(key) ?? { key, name: other, currency: bill.currency, net: 0, bills: [] };
      entry.name = other;
      entry.net += signed;
      entry.bills.push({ id: bill.id, title: bill.title, createdAt: bill.createdAt, amount: signed, dueDate: bill.settlement?.dueDate });
      byPerson.set(key, entry);
    }
  }
  return [...byPerson.values()].filter((p) => p.net !== 0).sort((a, b) => Math.abs(b.net) - Math.abs(a.net) || a.name.localeCompare(b.name));
}

/** "You're owed" / "you owe" totals for a header, per currency of the first balance (the common single-currency case). */
export function balanceTotals(list: PersonBalance[]): { owedToYou: Cents; youOwe: Cents } {
  return {
    owedToYou: sum(list.filter((p) => p.net > 0).map((p) => p.net)),
    youOwe: sum(list.filter((p) => p.net < 0).map((p) => -p.net)), // not -sum(): an empty list would give -0
  };
}

// ---- who is ahead and who is behind (for showing it) -----------------------------------------

export interface Position {
  name: string;
  /** What they owe for the bill. */
  share: Cents;
  /** What they paid up front. */
  fronted: Cents;
  /** fronted - share: positive = they are owed money back, negative = they still owe. */
  net: Cents;
}

export function positions(shares: Share[], paidBy: Settlement["paidBy"]): Position[] {
  const fronted = new Map<string, number>();
  for (const p of paidBy) fronted.set(p.name, (fronted.get(p.name) ?? 0) + p.amount);
  return shares.map((s) => {
    const f = fronted.get(s.name) ?? 0;
    return { name: s.name, share: s.amount, fronted: f, net: f - s.amount };
  });
}

/** The line under a person on the settlement list ("Paid RM60.00 · gets RM30.00 back"), or "" for nothing to say. */
export function positionNote(p: Position, currency: string): string {
  const m = (c: Cents) => formatMoney(c, currency);
  if (p.fronted > 0 && p.net > 0) return `Paid ${m(p.fronted)} · gets ${m(p.net)} back`;
  if (p.fronted > 0 && p.net < 0) return `Paid ${m(p.fronted)} · still owes ${m(-p.net)}`;
  if (p.fronted > 0) return "Paid their own share";
  if (p.net < 0) return `Owes ${m(-p.net)}`;
  return "";
}

/** Everyone paid exactly their own share (separate cards at the till): nobody owes anybody. */
export const everyonePaidOwn = (shares: Share[]): Settlement["paidBy"] =>
  shares.filter((s) => s.amount > 0).map((s) => ({ name: s.name, amount: s.amount }));

/**
 * The bill total changed under several payers (the split was redone): keep everyone's proportion of
 * what was fronted, scaled to the new total to the cent, instead of leaving amounts that no longer add up.
 */
export function scalePaidBy(paidBy: Settlement["paidBy"], total: Cents): Settlement["paidBy"] {
  const weights = paidBy.map((p) => p.amount);
  if (sum(weights) <= 0) return paidBy;
  const scaled = apportion(total, weights);
  return paidBy.map((p, i) => ({ name: p.name, amount: scaled[i] })).filter((p) => p.amount > 0);
}

// ---- messages -------------------------------------------------------------------------------

const clip = (s: string, n = 90) => (s.length > n ? `${s.slice(0, n - 1).trimEnd()}…` : s);

/**
 * What one person still owes, in one message however many people they owe. It says what they owe TO
 * WHOM, not "your share": with several payers the amount they owe is not their share (they may have
 * paid part of it, or owe two people). `to` is "me" or the creditor's name.
 */
export function nudgeText(o: {
  debtor: string;
  title: string;
  date: string | number | Date;
  currency: string;
  owes: { to: string; amount: Cents }[];
  /** Only pass this when what they owe is their whole share, so it describes what the money is for. */
  items?: string;
}): string {
  const head = `Hi ${o.debtor}! For *${o.title}* (${shortDate(o.date)}) you owe`;
  const tail = "Please pay when you can. Thanks! — via Bill.a";
  if (o.owes.length === 1) {
    // Only real item names help the reader; "33.3% of the bill" / "(nothing assigned)" are just noise here.
    const useful = !!o.items && o.items !== "(nothing assigned)" && !/of the bill$|the remaining|amount you gave|parts$/.test(o.items);
    const what = useful ? ` (${clip(o.items!)})` : "";
    return `${head} ${o.owes[0].to} *${formatMoney(o.owes[0].amount, o.currency)}*${what}. ${tail}`;
  }
  const list = o.owes.map((x) => `• ${x.to}: ${formatMoney(x.amount, o.currency)}`).join("\n");
  return `${head}:\n${list}\nTotal *${formatMoney(sum(o.owes.map((x) => x.amount)), o.currency)}*. ${tail}`;
}

/**
 * Everyone who still owes on one bill, for the group chat. One creditor: "still to pay <them>" with a
 * line per person. Several creditors: "who still owes what" with "Casey → Alex" lines.
 */
export function groupNudgeText(o: {
  title: string;
  date: string | number | Date;
  currency: string;
  /** Set when everyone owes the same person. */
  payTo?: string;
  lines: { debtor: string; to?: string; amount: Cents }[];
}): string {
  const money = (c: Cents) => formatMoney(c, o.currency);
  const head = `*${o.title}* (${shortDate(o.date)}) — ${o.payTo ? `still to pay ${o.payTo}` : "who still owes what"}:`;
  const list = o.lines.map((l) => (o.payTo ? `• ${l.debtor}: ${money(l.amount)}` : `• ${l.debtor} → ${l.to}: ${money(l.amount)}`)).join("\n");
  return `${head}\n${list}\nThanks! — via Bill.a`;
}

/** Everything one person still owes you across bills. */
export function balanceNudgeText(o: {
  name: string;
  payTo: string;
  currency: string;
  lines: { title: string; date: string | number | Date; amount: Cents }[];
}): string {
  const list = o.lines.map((l) => `• ${l.title} (${shortDate(l.date)}): ${formatMoney(l.amount, o.currency)}`).join("\n");
  const total = sum(o.lines.map((l) => l.amount));
  return `Hi ${o.name}! A reminder of what's still open:\n${list}\nTotal *${formatMoney(total, o.currency)}* to ${o.payTo}. Thanks! — via Bill.a`;
}

/** The summary people share: each share, who paid (one or several), and who still has to pay whom. */
export function summaryText(o: { currency: string; shares: Share[]; settlement?: Settlement | null; view?: SettleView | null }): string {
  const money = (c: Cents) => formatMoney(c, o.currency);
  const lines = [`*Bill-a Summary (${o.currency})*`, ""];
  for (const s of o.shares) lines.push(`👤 *${s.name}*: ${money(s.amount)}`);

  const payers = (o.settlement?.paidBy ?? []).filter((p) => p.amount > 0);
  if (payers.length === 1) lines.push("", `Paid by ${payers[0].name}.`);
  else if (payers.length > 1) lines.push("", `Paid by ${payers.map((p) => `${p.name} (${money(p.amount)})`).join(", ")}.`);

  if (o.view && o.view.ok && o.view.transfers.length > 0) {
    const open = o.view.transfers.filter((t) => !t.paid);
    if (open.length > 0) {
      lines.push("", "Still to pay:");
      for (const t of open) lines.push(`• ${t.from} → ${t.to}: ${money(t.amount)}`);
    } else {
      lines.push("", "All settled ✓");
    }
  }
  return lines.join("\n");
}

// ---- defaults and keeping a settlement in step with the shares -------------------------------

/** Nobody has been asked yet: the user (if we can tell who that is), else the first person, fronted it all. */
export function defaultSettlement(shares: Share[], me?: string): Settlement {
  const total = sum(shares.map((s) => s.amount));
  const payer = me && shares.some((s) => s.name === me) ? me : shares[0]?.name ?? "";
  return { ...(me ? { me } : {}), paidBy: singlePayer(payer, total), paid: {} };
}

/**
 * The shares changed (adjusted, cash-rounded, someone added or removed): a single payer always fronted
 * the whole bill, so follow it; if they left the group, the first person takes over. Several payers are
 * left as entered (a mismatch is shown, not silently fixed). Stale paid marks are the view's job.
 */
export function normalizeSettlement(s: Settlement, shares: Share[]): Settlement {
  if (shares.length === 0) return s;
  const total = sum(shares.map((x) => x.amount));
  const names = new Set(shares.map((x) => x.name));
  const me = s.me && names.has(s.me) ? s.me : undefined;
  const base = { ...s, me };
  if (s.paidBy.length === 1) {
    const payer = names.has(s.paidBy[0].name) ? s.paidBy[0].name : shares[0].name;
    return { ...base, paidBy: singlePayer(payer, total) };
  }
  return base;
}

// ---- the History list: a small summary per bill ---------------------------------------------

/** settled / open / due today / overdue: one rule, used by the bill view and the History list alike. */
export function settleStatus(outstanding: Cents, dueDate: string | undefined, today: string): { status: SettleStatus; dueIn: number | null } {
  const dueIn = dueDate ? daysUntil(dueDate, today) : null;
  const status: SettleStatus = outstanding === 0 ? "settled" : dueIn === null ? "open" : dueIn < 0 ? "overdue" : dueIn === 0 ? "due" : "open";
  return { status, dueIn };
}

export interface BillSettleSummary {
  owed: Cents;
  collected: Cents;
  outstanding: Cents;
  paidCount: number;
  transferCount: number;
  dueDate?: string;
  /** Who fronted it: one name, or several. */
  payers: string[];
  /** The user's own side (needs `settlement.me`): still owed to them, and still owed by them. */
  mine: { owedToYou: Cents; youOwe: Cents } | null;
  /** What was paid no longer matches the bill (the split changed): shown as "needs attention". */
  mismatch: boolean;
}

/**
 * Everything the History list needs about one bill's settle-up, without shipping the bill itself.
 * Date-dependent status is NOT here (the server's "today" is not the user's): the client applies
 * settleStatus() with its own date.
 */
export function summarizeSettlement(shares: Share[] | null | undefined, settlement: Settlement | null | undefined): BillSettleSummary | null {
  if (!shares || !settlement) return null;
  const view = settleView(shares, settlement, "1970-01-01");
  if (!view) return null;
  const payers = settlement.paidBy.filter((p) => p.amount > 0).map((p) => p.name);
  if (!view.ok) {
    return { owed: view.owed, collected: 0, outstanding: view.owed, paidCount: 0, transferCount: 0, dueDate: settlement.dueDate, payers, mine: null, mismatch: true };
  }

  const me = settlement.me ? norm(settlement.me) : null;
  let owedToYou = 0;
  let youOwe = 0;
  if (me) {
    for (const t of view.transfers) {
      if (t.paid) continue;
      if (norm(t.to) === me) owedToYou += t.amount;
      else if (norm(t.from) === me) youOwe += t.amount;
    }
  }
  return {
    owed: view.owed,
    collected: view.collected,
    outstanding: view.outstanding,
    paidCount: view.paidCount,
    transferCount: view.transfers.length,
    dueDate: settlement.dueDate,
    payers,
    mine: me ? { owedToYou, youOwe } : null,
    mismatch: false,
  };
}
