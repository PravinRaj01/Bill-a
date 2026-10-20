"use server";

import { z } from "zod";
import { getDb } from "@/lib/db/client";
import * as q from "@/lib/db/queries";
import { getUserIdOrNull, requireUserId } from "@/lib/auth/session";
import { localDate, summarizeSettlement, type PersonBalance } from "@/lib/split/settle";
import { computeBalances } from "@/lib/db/balances";

const idSchema = z.uuid();

/**
 * The user's bills for the History list, each with a compact settle-up summary (null when the bill
 * isn't tracked). The raw split and settlement are dropped here, so only the summary reaches the browser.
 */
export async function listBills() {
  const userId = await getUserIdOrNull();
  if (!userId) return [];
  const rows = await q.listBills(getDb(), userId);
  return rows.map(({ split, settlement, participants, ...rest }) => ({
    ...rest,
    // Who was in it (names only), for avatars on Home and History.
    people: (split ?? []).map((x) => x.name),
    // The friends' accounts on it (the user's own friends), so a Pals row finds its bills however they were named.
    linkedIds: (participants ?? []).flatMap((p) => (p.userId ? [p.userId] : [])),
    // Split for others: it knows its participants and the user isn't one of them. (The participants themselves
    // never reach the list: only this flag.)
    hosted: !!participants && !participants.some((p) => p.me),
    settle: summarizeSettlement(split, settlement),
  }));
}

export type BillListEntry = Awaited<ReturnType<typeof listBills>>[number];

export interface BalancesResult {
  /** What each person owes you (or you them) across tracked bills, biggest first. */
  people: PersonBalance[];
  /** How many bills are tracked at all. */
  tracked: number;
  /** Older tracked bills that don't say which person is you: they can't count towards your balances. (Bills split for others are not counted here: they were never yours.) */
  withoutMe: number;
}

/** Net balances across the signed-in user's tracked bills. Guests have none. */
export async function getBalances(): Promise<BalancesResult> {
  const userId = await getUserIdOrNull();
  if (!userId) return { people: [], tracked: 0, withoutMe: 0 };
  // Friends' bills shared with me count too, but only what runs between me and the friend who owns each one.
  return computeBalances(getDb(), userId, localDate());
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

/** "Session N" numbering: computed server-side from the caller's own count. */
export async function nextSessionTitle() {
  const userId = await getUserIdOrNull();
  if (!userId) return "Session 1";
  const { count } = await q.billStats(getDb(), userId);
  return `Session ${count + 1}`;
}
