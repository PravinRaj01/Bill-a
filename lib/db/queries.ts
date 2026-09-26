import { and, desc, eq, inArray, sql } from "drizzle-orm";
import type { Db } from "./client";
import { billHistory, savedGroups } from "./schema";
import type { BillData } from "@/types/domain";

// THE AUTHORIZATION LAYER — this file is what replaces Supabase RLS.
//
// Rule, without exceptions: every function here takes `userId` as a required
// parameter and puts `eq(table.userId, userId)` in its WHERE clause. There is
// deliberately NO function that reads or writes saved_groups / bill_history
// without one, so "forgot the user filter" can't be written by accident — the
// six queries that were IDOR holes before (bare select('*'), delete by id)
// simply have no equivalent here.
//
// `userId` must come from the verified server-side session (lib/actions/*),
// NEVER from client input. lib/db/queries.test.ts proves user A cannot read,
// change or delete user B's rows through any of these.

// --- saved groups ----------------------------------------------------------

export const listGroups = (db: Db, userId: string) =>
  db
    .select()
    .from(savedGroups)
    .where(eq(savedGroups.userId, userId))
    .orderBy(desc(savedGroups.createdAt));

export async function getGroup(db: Db, userId: string, id: string) {
  const [row] = await db
    .select()
    .from(savedGroups)
    .where(and(eq(savedGroups.id, id), eq(savedGroups.userId, userId)))
    .limit(1);
  return row ?? null;
}

/**
 * Saves a group. Saving a name the user already has (ignoring case and surrounding
 * spaces) UPDATES that group's members instead of creating a lookalike — the
 * (user_id, group_key) unique constraint makes that race-proof, so pressing "Start
 * Scanning" twice, or going Back and forward, can't produce "Best Couple" x2. The
 * existing group keeps its original display name and id.
 */
export async function createGroup(
  db: Db,
  userId: string,
  input: { groupName: string; names: string[] },
) {
  const [row] = await db
    .insert(savedGroups)
    .values({ userId, groupName: input.groupName, names: input.names })
    .onConflictDoUpdate({ target: [savedGroups.userId, savedGroups.groupKey], set: { names: input.names } })
    .returning();
  return row;
}

export async function updateGroupNames(db: Db, userId: string, id: string, names: string[]) {
  const [row] = await db
    .update(savedGroups)
    .set({ names })
    .where(and(eq(savedGroups.id, id), eq(savedGroups.userId, userId)))
    .returning();
  return row ?? null;
}

/** True if a row was deleted; false if it didn't exist OR belongs to someone else. */
export async function deleteGroup(db: Db, userId: string, id: string) {
  const rows = await db
    .delete(savedGroups)
    .where(and(eq(savedGroups.id, id), eq(savedGroups.userId, userId)))
    .returning({ id: savedGroups.id });
  return rows.length > 0;
}

// --- bill history ----------------------------------------------------------

// The list view never needs the (large) jsonb `data`, so it isn't selected.
const listColumns = {
  id: billHistory.id,
  billTitle: billHistory.billTitle,
  merchantCategory: billHistory.merchantCategory,
  totalAmount: billHistory.totalAmount,
  currency: billHistory.currency,
  providerTier: billHistory.providerTier,
  createdAt: billHistory.createdAt,
};

export const listBills = (db: Db, userId: string) =>
  db
    .select(listColumns)
    .from(billHistory)
    .where(eq(billHistory.userId, userId))
    .orderBy(desc(billHistory.createdAt));

export type BillListItem = Awaited<ReturnType<typeof listBills>>[number];

export async function getBill(db: Db, userId: string, id: string) {
  const [row] = await db
    .select()
    .from(billHistory)
    .where(and(eq(billHistory.id, id), eq(billHistory.userId, userId)))
    .limit(1);
  return row ?? null;
}

/** Returns how many rows were actually deleted — ids owned by others are silently ignored. */
export async function deleteBills(db: Db, userId: string, ids: string[]) {
  if (ids.length === 0) return 0;
  const rows = await db
    .delete(billHistory)
    .where(and(eq(billHistory.userId, userId), inArray(billHistory.id, ids)))
    .returning({ id: billHistory.id });
  return rows.length;
}

export async function clearBills(db: Db, userId: string) {
  const rows = await db
    .delete(billHistory)
    .where(eq(billHistory.userId, userId))
    .returning({ id: billHistory.id });
  return rows.length;
}

export async function billStats(db: Db, userId: string) {
  const [row] = await db
    .select({
      count: sql<number>`count(*)::int`,
      totalCents: sql<number>`coalesce(sum(${billHistory.totalAmount}), 0)::bigint`.mapWith(Number),
    })
    .from(billHistory)
    .where(eq(billHistory.userId, userId));
  return { count: row?.count ?? 0, totalCents: row?.totalCents ?? 0 };
}

export interface BillInput {
  clientId: string;
  billTitle: string;
  merchantCategory?: string | null;
  totalAmount: number; // cents
  currency: string;
  data: BillData;
  providerTier?: string | null;
}

/**
 * Idempotent save, keyed on (user_id, client_id). Retrying the same client_id
 * updates the one row instead of inserting a duplicate — this is what makes the
 * offline outbox safe to retry. The conflict target includes user_id, so a
 * client_id that happens to match ANOTHER user's row can never overwrite it: it
 * just creates this user's own row.
 */
export async function upsertBill(db: Db, userId: string, input: BillInput) {
  const values = {
    userId,
    clientId: input.clientId,
    billTitle: input.billTitle,
    merchantCategory: input.merchantCategory ?? null,
    totalAmount: input.totalAmount,
    currency: input.currency,
    data: input.data,
    providerTier: input.providerTier ?? null,
  };
  const [row] = await db
    .insert(billHistory)
    .values(values)
    .onConflictDoUpdate({
      target: [billHistory.userId, billHistory.clientId],
      set: {
        billTitle: values.billTitle,
        merchantCategory: values.merchantCategory,
        totalAmount: values.totalAmount,
        currency: values.currency,
        data: values.data,
        providerTier: values.providerTier,
      },
    })
    .returning({ id: billHistory.id, clientId: billHistory.clientId });
  return row;
}
