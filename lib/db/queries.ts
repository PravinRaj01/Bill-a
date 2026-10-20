import { and, desc, eq, inArray, isNotNull, isNull, sql } from "drizzle-orm";
import type { Db } from "./client";
import { billHistory, billShares, people, savedGroups, users } from "./schema";
import { notify } from "./notifications";
import { parsePrefs, type Category, type NotifyPrefs } from "@/lib/notifications/prefs";
import { propagateRename } from "./rename";
import { confirmSatisfiedClaims, syncShares, withFriendIds } from "./social";
import type { BillData, Participant, PaymentDetails, Settlement, SplitRecord } from "@/types/domain";

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
  input: { groupName: string; names: string[]; includesMe?: boolean },
) {
  const includesMe = input.includesMe ?? true;
  const [row] = await db
    .insert(savedGroups)
    .values({ userId, groupName: input.groupName, names: input.names, includesMe })
    .onConflictDoUpdate({ target: [savedGroups.userId, savedGroups.groupKey], set: { names: input.names, includesMe } })
    .returning();
  return row;
}

export async function updateGroupNames(db: Db, userId: string, id: string, names: string[], includesMe?: boolean) {
  const [row] = await db
    .update(savedGroups)
    .set({ names, ...(includesMe === undefined ? {} : { includesMe }) })
    .where(and(eq(savedGroups.id, id), eq(savedGroups.userId, userId)))
    .returning();
  return row ?? null;
}

export type UpdateGroupResult = { ok: true; group: typeof savedGroups.$inferSelect } | { ok: false; reason: "not-found" | "name-taken" };

/**
 * Renames a group and/or replaces its members, for the signed-in user's own group only. The new name may
 * differ from the old one only in case or spacing (same group), but not match ANOTHER of the user's groups:
 * the (user_id, group_key) unique constraint decides that, and it is reported as "name-taken" rather than
 * thrown, so the screen can say so.
 */
export async function updateGroup(
  db: Db,
  userId: string,
  id: string,
  input: { groupName: string; names: string[]; includesMe?: boolean },
): Promise<UpdateGroupResult> {
  try {
    const [row] = await db
      .update(savedGroups)
      .set({ groupName: input.groupName, names: input.names, ...(input.includesMe === undefined ? {} : { includesMe: input.includesMe }) })
      .where(and(eq(savedGroups.id, id), eq(savedGroups.userId, userId)))
      .returning();
    return row ? { ok: true, group: row } : { ok: false, reason: "not-found" };
  } catch (e) {
    const err = e as { code?: string; cause?: { code?: string }; message?: string };
    const code = err.code ?? err.cause?.code;
    if (code === "23505" || /duplicate key|unique constraint/i.test(err.message ?? "")) return { ok: false, reason: "name-taken" };
    throw e;
  }
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
  // Two small slices of the jsonb for the list's settle-up status; never the whole bill.
  split: sql<SplitRecord[] | null>`${billHistory.data} -> 'split'`,
  settlement: sql<Settlement | null>`${billHistory.data} -> 'settlement'`,
  // Who was in it and which one is the user (null on bills saved before batch 5).
  participants: sql<Participant[] | null>`${billHistory.data} -> 'participants'`,
};

export const listBills = (db: Db, userId: string) =>
  db
    .select(listColumns)
    .from(billHistory)
    .where(and(eq(billHistory.userId, userId), isNull(billHistory.hiddenAt)))
    .orderBy(desc(billHistory.createdAt));

/** Bills the owner hid from their own History (a duplicate of a friend's), newest first. */
export const listHiddenBills = (db: Db, userId: string) =>
  db
    .select({ id: billHistory.id, billTitle: billHistory.billTitle, totalAmount: billHistory.totalAmount, currency: billHistory.currency, createdAt: billHistory.createdAt })
    .from(billHistory)
    .where(and(eq(billHistory.userId, userId), isNotNull(billHistory.hiddenAt)))
    .orderBy(desc(billHistory.createdAt));

/** Hide (or bring back) one of MY bills. False when it isn't mine. Hidden bills leave History and balances; friends keep their copy. */
export async function setBillHidden(db: Db, userId: string, id: string, hidden: boolean): Promise<boolean> {
  const rows = await db
    .update(billHistory)
    .set({ hiddenAt: hidden ? sql`now()` : null })
    .where(and(eq(billHistory.id, id), eq(billHistory.userId, userId)))
    .returning({ id: billHistory.id });
  return rows.length > 0;
}

export type BillListItem = Awaited<ReturnType<typeof listBills>>[number];

/**
 * Only the bills whose settle-up is being tracked, with just what balances need (title, currency, when,
 * the split and the settlement), newest first. Never the whole bill.
 */
export const listTrackedBills = (db: Db, userId: string) =>
  db
    .select({
      id: billHistory.id,
      billTitle: billHistory.billTitle,
      currency: billHistory.currency,
      createdAt: billHistory.createdAt,
      split: sql<SplitRecord[] | null>`${billHistory.data} -> 'split'`,
      settlement: sql<Settlement | null>`${billHistory.data} -> 'settlement'`,
      participants: sql<Participant[] | null>`${billHistory.data} -> 'participants'`,
    })
    .from(billHistory)
    .where(and(eq(billHistory.userId, userId), isNull(billHistory.hiddenAt), sql`${billHistory.data} -> 'settlement' is not null`))
    .orderBy(desc(billHistory.createdAt));

export async function getBill(db: Db, userId: string, id: string) {
  const [row] = await db
    .select()
    .from(billHistory)
    .where(and(eq(billHistory.id, id), eq(billHistory.userId, userId)))
    .limit(1);
  return row ?? null;
}

/** Friends a bill was shared with, read BEFORE the bill goes (the share rows cascade away with it). */
async function sharesOf(db: Db, ownerId: string, ids: string[] | null) {
  return db
    .select({ billId: billHistory.id, title: billHistory.billTitle, friendId: billShares.userId })
    .from(billShares)
    .innerJoin(billHistory, eq(billHistory.id, billShares.billId))
    .where(and(eq(billHistory.userId, ownerId), ids ? inArray(billHistory.id, ids) : undefined));
}

/** "Alex deleted 'Dinner out'", to everyone a deleted bill was shared with. The bill is gone, so no bill id. */
async function tellDeleted(db: Db, ownerId: string, shares: Awaited<ReturnType<typeof sharesOf>>, deleted: string[]) {
  const gone = new Set(deleted);
  for (const sh of shares.filter((x) => gone.has(x.billId))) {
    await notify(db, { userId: sh.friendId, kind: "bill_deleted", actorId: ownerId, payload: { title: sh.title }, dedupe: `bill_deleted:${sh.billId}` });
  }
}

/** Returns how many rows were actually deleted — ids owned by others are silently ignored. */
export async function deleteBills(db: Db, userId: string, ids: string[]) {
  if (ids.length === 0) return 0;
  const shares = await sharesOf(db, userId, ids);
  const rows = await db
    .delete(billHistory)
    .where(and(eq(billHistory.userId, userId), inArray(billHistory.id, ids)))
    .returning({ id: billHistory.id });
  await tellDeleted(db, userId, shares, rows.map((r) => r.id));
  return rows.length;
}

export async function clearBills(db: Db, userId: string) {
  const shares = await sharesOf(db, userId, null);
  const rows = await db
    .delete(billHistory)
    .where(eq(billHistory.userId, userId))
    .returning({ id: billHistory.id });
  await tellDeleted(db, userId, shares, rows.map((r) => r.id));
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

/** Total of all the user's bills per currency, in cents (bills can be in different currencies, so they are never summed together). */
export async function billTotalsByCurrency(db: Db, userId: string): Promise<{ currency: string; cents: number }[]> {
  return db
    .select({
      currency: billHistory.currency,
      cents: sql<number>`coalesce(sum(${billHistory.totalAmount}), 0)::bigint`.mapWith(Number),
    })
    .from(billHistory)
    .where(eq(billHistory.userId, userId))
    .groupBy(billHistory.currency)
    .orderBy(desc(sql`sum(${billHistory.totalAmount})`));
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
  // Who was on this bill before, so a friend the owner left out of a past bill isn't added back by a re-save.
  const [prev] = await db
    .select({ split: sql<SplitRecord[] | null>`${billHistory.data} -> 'split'` })
    .from(billHistory)
    .where(and(eq(billHistory.userId, userId), eq(billHistory.clientId, input.clientId)))
    .limit(1);
  const previousNames = prev ? (prev.split ?? []).map((s) => s.name.trim().toLowerCase()) : null;
  // The client says who each name stands for; only accepted friends stay linked.
  const data = await withFriendIds(db, userId, input.data);
  const values = {
    userId,
    clientId: input.clientId,
    billTitle: input.billTitle,
    merchantCategory: input.merchantCategory ?? null,
    totalAmount: input.totalAmount,
    currency: input.currency,
    data,
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
        updatedAt: sql`now()`,
      },
    })
    .returning({ id: billHistory.id, clientId: billHistory.clientId });
  // Friends on the bill get it in their History; a paid mark that answers an "I've paid" confirms it.
  await syncShares(db, userId, row.id, data, previousNames);
  await confirmSatisfiedClaims(db, row.id, data.settlement, data.split);
  return row;
}

// --- people ----------------------------------------------------------------
// Contacts the user splits with. Every function is keyed by the session's user id; there is no way to read or
// change another user's contacts through these.

export const listPeople = (db: Db, userId: string) => db.select().from(people).where(eq(people.userId, userId)).orderBy(people.nameKey);

/** The contacts, with the public side (name, @username, portrait) of those linked to a friend. Only this user's friends. */
export async function listPeopleWithFriends(db: Db, userId: string) {
  const rows = await listPeople(db, userId);
  const ids = rows.flatMap((r) => (r.linkedUserId ? [r.linkedUserId] : []));
  const found = ids.length ? await db.select({ id: users.id, name: users.name, username: users.username, avatarSeed: users.avatarSeed }).from(users).where(inArray(users.id, ids)) : [];
  const byId = new Map(found.map((u) => [u.id, u]));
  return rows.map((r) => {
    const u = r.linkedUserId ? byId.get(r.linkedUserId) : undefined;
    return { ...r, friend: u ? { name: u.name, username: u.username, avatarSeed: u.avatarSeed } : null };
  });
}

export type PersonInput = { name: string; phone?: string | null };

/**
 * Adds a contact, or UPDATES the one with the same name (ignoring case and spacing): the (user_id, name_key)
 * unique constraint makes that race-proof. A phone is only overwritten when one is given.
 */
export async function upsertPerson(db: Db, userId: string, input: PersonInput) {
  const phone = input.phone ?? null;
  const [row] = await db
    .insert(people)
    .values({ userId, name: input.name, phone })
    .onConflictDoUpdate({
      target: [people.userId, people.nameKey],
      set: { ...(phone ? { phone } : {}), updatedAt: sql`now()` },
    })
    .returning();
  return row;
}

export type UpdatePersonResult = { ok: true; person: typeof people.$inferSelect } | { ok: false; reason: "not-found" | "name-taken" };

/** Renames a contact and/or sets (or clears with null) their number. A rename onto another contact is "name-taken". */
export async function updatePerson(db: Db, userId: string, id: string, patch: { name?: string; phone?: string | null }): Promise<UpdatePersonResult> {
  const set: { name?: string; phone?: string | null; updatedAt: ReturnType<typeof sql> } = { updatedAt: sql`now()` };
  if (patch.name !== undefined) set.name = patch.name;
  if (patch.phone !== undefined) set.phone = patch.phone;
  try {
    const [row] = await db.update(people).set(set).where(and(eq(people.id, id), eq(people.userId, userId))).returning();
    return row ? { ok: true, person: row } : { ok: false, reason: "not-found" };
  } catch (e) {
    const err = e as { code?: string; cause?: { code?: string }; message?: string };
    const code = err.code ?? err.cause?.code;
    if (code === "23505" || /duplicate key|unique constraint/i.test(err.message ?? "")) return { ok: false, reason: "name-taken" };
    throw e;
  }
}

/** True if a row was deleted; false if it didn't exist OR belongs to someone else. */
export async function deletePerson(db: Db, userId: string, id: string) {
  const rows = await db.delete(people).where(and(eq(people.id, id), eq(people.userId, userId))).returning({ id: people.id });
  return rows.length > 0;
}

export interface Suggestion {
  name: string;
  /** In how many of the user's bills and groups this name appears. */
  count: number;
}

/**
 * Names the user has split with (in saved bills and groups) who aren't contacts yet, most frequent first.
 * `excludeNames` are left out too (the user's own name). Names are matched ignoring case and spacing, and the
 * spelling shown is the most common one.
 */
export async function suggestPeople(db: Db, userId: string, excludeNames: string[], limit = 12): Promise<Suggestion[]> {
  const exclude = excludeNames.map((n) => n.trim().toLowerCase()).filter(Boolean);
  const rows = await db.execute(sql`
    with seen as (
      select btrim(e->>'name') as name
        from bill_history b, jsonb_array_elements(coalesce(b.data -> 'split', '[]'::jsonb)) e
       where b.user_id = ${userId}
      union all
      select btrim(n) as name
        from saved_groups g, jsonb_array_elements_text(g.names) n
       where g.user_id = ${userId}
    ), mine as (
      -- however the user was called on their own bills (the name may have changed since): never suggest yourself
      select lower(btrim(b.data -> 'settlement' ->> 'me')) as key from bill_history b where b.user_id = ${userId}
      union
      select lower(btrim(p ->> 'name')) from bill_history b, jsonb_array_elements(coalesce(b.data -> 'participants', '[]'::jsonb)) p
       where b.user_id = ${userId} and p ->> 'me' = 'true'
    ), counted as (
      select lower(name) as key, name, count(*) as n from seen where name <> '' group by lower(name), name
    ), best as (
      select key, (array_agg(name order by n desc, name))[1] as name, sum(n)::int as count from counted group by key
    )
    select name, count from best
     where key not in (select jsonb_array_elements_text(${JSON.stringify(exclude)}::jsonb))
       and key not in (select name_key from people where user_id = ${userId})
       and key not in (select key from mine where key is not null)
     order by count desc, name
     limit ${limit}
  `);
  const list = (Array.isArray(rows) ? rows : (rows as { rows?: unknown[] }).rows ?? []) as { name: string; count: number }[];
  return list.map((r) => ({ name: r.name, count: Number(r.count) }));
}

// --- profile ---------------------------------------------------------------

export interface Profile {
  username: string | null;
  avatarSeed: string | null;
  tz: string;
}

export async function getProfile(db: Db, userId: string): Promise<Profile | null> {
  const [row] = await db.select({ username: users.username, avatarSeed: users.avatarSeed, tz: users.tz }).from(users).where(eq(users.id, userId)).limit(1);
  return row ?? null;
}

/**
 * Sets the user's username and/or portrait (null clears, undefined leaves alone), for the session's user only.
 * A username that is already taken comes back as "username-taken" (the unique constraint decides, so two
 * people claiming the same name at once can't both win) rather than being thrown.
 */
export async function setProfile(
  db: Db,
  userId: string,
  patch: { username?: string | null; avatarSeed?: string | null },
): Promise<{ ok: true; profile: Profile } | { ok: false; reason: "username-taken" | "not-found" }> {
  const set: { username?: string | null; avatarSeed?: string | null } = {};
  if (patch.username !== undefined) set.username = patch.username;
  if (patch.avatarSeed !== undefined) set.avatarSeed = patch.avatarSeed;
  try {
    if (Object.keys(set).length > 0) {
      const rows = await db.update(users).set(set).where(eq(users.id, userId)).returning({ id: users.id });
      if (rows.length === 0) return { ok: false, reason: "not-found" };
    }
  } catch (e) {
    const err = e as { code?: string; cause?: { code?: string }; message?: string };
    const code = err.code ?? err.cause?.code;
    if (code === "23505" || /duplicate key|unique constraint/i.test(err.message ?? "")) return { ok: false, reason: "username-taken" };
    throw e;
  }
  const profile = await getProfile(db, userId);
  return profile ? { ok: true, profile } : { ok: false, reason: "not-found" };
}

export async function setTimeZone(db: Db, userId: string, tz: string): Promise<void> {
  await db.update(users).set({ tz }).where(eq(users.id, userId));
}

/** The user's notification preferences (only categories switched off are stored). */
export async function getNotifyPrefs(db: Db, userId: string): Promise<NotifyPrefs> {
  const [row] = await db.select({ p: users.notifyPrefs }).from(users).where(eq(users.id, userId)).limit(1);
  return parsePrefs(row?.p);
}

/** Switch one category on or off for push. Keyed by the session's user id only. */
export async function setNotifyPref(db: Db, userId: string, category: Category, on: boolean): Promise<NotifyPrefs> {
  await db
    .update(users)
    .set({ notifyPrefs: on ? sql`${users.notifyPrefs} - ${category}::text` : sql`${users.notifyPrefs} || ${JSON.stringify({ [category]: false })}::jsonb` })
    .where(eq(users.id, userId));
  return getNotifyPrefs(db, userId);
}

/**
 * The user's display name (null clears it). Keyed by the session's user id only. Friends' bills, contacts and
 * groups that name them by their old name follow the change (lib/db/rename.ts).
 */
export async function setUserName(db: Db, userId: string, name: string | null): Promise<string | null> {
  const [before] = await db.select({ name: users.name }).from(users).where(eq(users.id, userId)).limit(1);
  await db.update(users).set({ name }).where(eq(users.id, userId));
  await propagateRename(db, userId, before?.name ?? null, name);
  return name;
}

// --- payment details (how friends pay this user back) -----------------------
// Lives on the user row; both functions are keyed by the session's userId only.
// `details` must already have been through parsePaymentDetails (lib/validation/payment.ts).

export async function getPaymentDetails(db: Db, userId: string): Promise<PaymentDetails | null> {
  const [row] = await db.select({ paymentDetails: users.paymentDetails }).from(users).where(eq(users.id, userId)).limit(1);
  return row?.paymentDetails ?? null;
}

/** An empty object clears the details (stored as null). */
export async function setPaymentDetails(db: Db, userId: string, details: PaymentDetails): Promise<PaymentDetails | null> {
  const value = Object.keys(details).length ? details : null;
  await db.update(users).set({ paymentDetails: value }).where(eq(users.id, userId));
  return value;
}
