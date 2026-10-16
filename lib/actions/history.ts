"use server";

import { z } from "zod";
import { getDb } from "@/lib/db/client";
import * as q from "@/lib/db/queries";
import { getUserIdOrNull, requireUserId } from "@/lib/auth/session";
import { aggregateBalances, localDate, summarizeSettlement, type PersonBalance } from "@/lib/split/settle";

const idSchema = z.uuid();

/**
 * The user's bills for the History list, each with a compact settle-up summary (null when the bill
 * isn't tracked). The raw split and settlement are dropped here, so only the summary reaches the browser.
 */
export async function listBills() {
  const userId = await getUserIdOrNull();
  if (!userId) return [];
  const rows = await q.listBills(getDb(), userId);
  return rows.map(({ split, settlement, ...rest }) => ({
    ...rest,
    // Who was in it (names only), for avatars on Home and History.
    people: (split ?? []).map((x) => x.name),
    settle: summarizeSettlement(split, settlement),
  }));
}

export type BillListEntry = Awaited<ReturnType<typeof listBills>>[number];

export interface BalancesResult {
  /** What each person owes you (or you them) across tracked bills, biggest first. */
  people: PersonBalance[];
  /** How many bills are tracked at all. */
  tracked: number;
  /** Tracked bills that don't say which person is you: they can't count towards your balances. */
  withoutMe: number;
}

/** Net balances across the signed-in user's tracked bills. Guests have none. */
export async function getBalances(): Promise<BalancesResult> {
  const userId = await getUserIdOrNull();
  if (!userId) return { people: [], tracked: 0, withoutMe: 0 };
  const rows = await q.listTrackedBills(getDb(), userId);
  const people = aggregateBalances(
    rows.map((r) => ({ id: r.id, title: r.billTitle, createdAt: r.createdAt, currency: r.currency, split: r.split ?? [], settlement: r.settlement ?? undefined })),
    localDate(),
  );
  return { people, tracked: rows.length, withoutMe: rows.filter((r) => !r.settlement?.me).length };
}

export async function getBill(id: string) {
  const userId = await getUserIdOrNull();
  if (!userId || !idSchema.safeParse(id).success) return null;
  return q.getBill(getDb(), userId, id);
}

export async function deleteBills(ids: string[]) {
  const userId = await requireUserId();
  return q.deleteBills(getDb(), userId, z.array(idSchema).max(500).parse(ids));
}

export async function clearHistory() {
  const userId = await requireUserId();
  return q.clearBills(getDb(), userId);
}

export async function getAccountStats() {
  const userId = await getUserIdOrNull();
  if (!userId) return { count: 0, totals: [] as { currency: string; cents: number }[] };
  const db = getDb();
  const [{ count }, totals] = await Promise.all([q.billStats(db, userId), q.billTotalsByCurrency(db, userId)]);
  return { count, totals };
}

/** "Session N" numbering: computed server-side from the caller's own count. */
export async function nextSessionTitle() {
  const userId = await getUserIdOrNull();
  if (!userId) return "Session 1";
  const { count } = await q.billStats(getDb(), userId);
  return `Session ${count + 1}`;
}
