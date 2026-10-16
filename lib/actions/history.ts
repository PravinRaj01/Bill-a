"use server";

import { z } from "zod";
import { getDb } from "@/lib/db/client";
import * as q from "@/lib/db/queries";
import { getUserIdOrNull, requireUserId } from "@/lib/auth/session";
import { summarizeSettlement } from "@/lib/split/settle";

const idSchema = z.uuid();

/**
 * The user's bills for the History list, each with a compact settle-up summary (null when the bill
 * isn't tracked). The raw split and settlement are dropped here, so only the summary reaches the browser.
 */
export async function listBills() {
  const userId = await getUserIdOrNull();
  if (!userId) return [];
  const rows = await q.listBills(getDb(), userId);
  return rows.map(({ split, settlement, ...rest }) => ({ ...rest, settle: summarizeSettlement(split, settlement) }));
}

export type BillListEntry = Awaited<ReturnType<typeof listBills>>[number];

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
  return userId ? q.billStats(getDb(), userId) : { count: 0, totalCents: 0 };
}

/** "Session N" numbering: computed server-side from the caller's own count. */
export async function nextSessionTitle() {
  const userId = await getUserIdOrNull();
  if (!userId) return "Session 1";
  const { count } = await q.billStats(getDb(), userId);
  return `Session ${count + 1}`;
}
