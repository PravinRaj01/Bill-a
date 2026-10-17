import { and, desc, eq, inArray, or, sql } from "drizzle-orm";

import { billMe } from "@/lib/split/participants";
import { settleView, localDate } from "@/lib/split/settle";
import type { BillData, Participant, Settlement, SplitRecord } from "@/types/domain";
import type { Db } from "./client";
import { billHistory, billShares, friendships, paymentClaims, people, users } from "./schema";

// Friends, shared bills and payment claims. Same rule as queries.ts: every function takes the session's
// userId and scopes by it, and what a friend can reach is limited to rows an accepted friendship or an
// explicit share grants. Nothing here ever returns an email, a password hash or another user's id unless
// that person is the caller's friend.

const MAX_PENDING_OUT = 50;

// (user_a, user_b) is stored smaller-id-first. Ordering is decided by the database (least/greatest) so it
// always agrees with the table's own CHECK, whatever the collation.
const pairWhere = (a: string, b: string) => and(eq(friendships.userA, sql`least(${a}::text, ${b}::text)`), eq(friendships.userB, sql`greatest(${a}::text, ${b}::text)`));

export interface PublicUser {
  id: string;
  username: string | null;
  name: string | null;
  avatarSeed: string | null;
}

const publicUser = { id: users.id, username: users.username, name: users.name, avatarSeed: users.avatarSeed };

/** Exact match on a claimed @username (no partial search, so accounts can't be browsed). */
export async function findUserByUsername(db: Db, username: string): Promise<PublicUser | null> {
  const [row] = await db.select(publicUser).from(users).where(eq(users.username, username.toLowerCase())).limit(1);
  return row ?? null;
}

export type Relation = "none" | "friends" | "outgoing" | "incoming";

export async function relationTo(db: Db, me: string, other: string): Promise<Relation> {
  const [row] = await db.select().from(friendships).where(pairWhere(me, other)).limit(1);
  if (!row) return "none";
  if (row.status === "accepted") return "friends";
  if (row.status === "pending") return row.requestedBy === me ? "outgoing" : "incoming";
  // Declined: silent to whoever asked; the one who declined may ask back later.
  return "none";
}

export async function areFriends(db: Db, a: string, b: string): Promise<boolean> {
  return (await relationTo(db, a, b)) === "friends";
}

export type RequestResult =
  | { ok: true; status: "requested" | "accepted" }
  | { ok: false; reason: "self" | "not-found" | "already-friends" | "already-requested" | "too-many" };

/**
 * Asks someone to be friends. A request that crosses an incoming one is accepted at once. A request that was
 * declined before looks exactly like a new one to the asker (nothing changes, nothing is revealed).
 */
export async function requestFriend(db: Db, me: string, otherId: string): Promise<RequestResult> {
  if (me === otherId) return { ok: false, reason: "self" };
  const [exists] = await db.select({ id: users.id }).from(users).where(eq(users.id, otherId)).limit(1);
  if (!exists) return { ok: false, reason: "not-found" };

  const [row] = await db.select().from(friendships).where(pairWhere(me, otherId)).limit(1);
  if (row) {
    if (row.status === "accepted") return { ok: false, reason: "already-friends" };
    if (row.status === "pending") {
      if (row.requestedBy === me) return { ok: false, reason: "already-requested" };
      await answerRequest(db, me, otherId, true); // they asked me first: this is a yes
      return { ok: true, status: "accepted" };
    }
    // declined
    if (row.requestedBy === me) return { ok: true, status: "requested" }; // stays declined, silently
    await db.update(friendships).set({ status: "pending", requestedBy: me, createdAt: sql`now()` }).where(pairWhere(me, otherId));
    return { ok: true, status: "requested" };
  }

  const [{ n }] = await db
    .select({ n: sql<number>`count(*)::int` })
    .from(friendships)
    .where(and(eq(friendships.requestedBy, me), eq(friendships.status, "pending")));
  if (n >= MAX_PENDING_OUT) return { ok: false, reason: "too-many" };

  await db
    .insert(friendships)
    .values({ userA: sql`least(${me}::text, ${otherId}::text)`, userB: sql`greatest(${me}::text, ${otherId}::text)`, requestedBy: me })
    .onConflictDoNothing();
  return { ok: true, status: "requested" };
}

/** Links (or creates) the contact for a friend in this user's People, so the friend shows up in pickers. */
async function linkContact(db: Db, userId: string, friend: PublicUser): Promise<void> {
  const already = await db.select({ id: people.id }).from(people).where(and(eq(people.userId, userId), eq(people.linkedUserId, friend.id))).limit(1);
  if (already.length) return;
  const display = (friend.name ?? "").trim() || friend.username || "Friend";
  const tryNames = [display, friend.username ?? `${display} 2`];
  for (const name of tryNames) {
    // An existing, unlinked contact with that name IS this friend: link it instead of adding a lookalike.
    const linked = await db
      .update(people)
      .set({ linkedUserId: friend.id, updatedAt: sql`now()` })
      .where(and(eq(people.userId, userId), eq(people.nameKey, name.trim().toLowerCase()), sql`${people.linkedUserId} is null`))
      .returning({ id: people.id });
    if (linked.length) return;
    const inserted = await db.insert(people).values({ userId, name, linkedUserId: friend.id }).onConflictDoNothing().returning({ id: people.id });
    if (inserted.length) return;
  }
}

/** Accept or decline a request somebody sent ME. False when there is no such pending request. */
export async function answerRequest(db: Db, me: string, otherId: string, accept: boolean): Promise<boolean> {
  const [row] = await db.select().from(friendships).where(and(pairWhere(me, otherId), eq(friendships.status, "pending"))).limit(1);
  if (!row || row.requestedBy === me) return false;
  await db
    .update(friendships)
    .set(accept ? { status: "accepted", acceptedAt: sql`now()` } : { status: "declined" })
    .where(pairWhere(me, otherId));
  if (accept) {
    const found = await db.select(publicUser).from(users).where(inArray(users.id, [me, otherId]));
    const mine = found.find((u) => u.id === me);
    const theirs = found.find((u) => u.id === otherId);
    if (mine && theirs) {
      await linkContact(db, me, theirs);
      await linkContact(db, otherId, mine);
    }
  }
  return true;
}

/** Ends a friendship (or withdraws a request): shares between the two stop at once, contacts become plain names. */
export async function removeFriend(db: Db, me: string, otherId: string): Promise<void> {
  await db.delete(friendships).where(pairWhere(me, otherId));
  await db.update(people).set({ linkedUserId: null, updatedAt: sql`now()` }).where(or(and(eq(people.userId, me), eq(people.linkedUserId, otherId)), and(eq(people.userId, otherId), eq(people.linkedUserId, me))));
  // Bills I own shared with them, and bills they own shared with me.
  await db.delete(billShares).where(and(eq(billShares.userId, otherId), inArray(billShares.billId, db.select({ id: billHistory.id }).from(billHistory).where(eq(billHistory.userId, me)))));
  await db.delete(billShares).where(and(eq(billShares.userId, me), inArray(billShares.billId, db.select({ id: billHistory.id }).from(billHistory).where(eq(billHistory.userId, otherId)))));
}

export interface FriendsState {
  friends: PublicUser[];
  incoming: PublicUser[];
  outgoing: PublicUser[];
}

export async function listFriendsState(db: Db, me: string): Promise<FriendsState> {
  const rows = await db
    .select({
      status: friendships.status,
      requestedBy: friendships.requestedBy,
      userA: friendships.userA,
      userB: friendships.userB,
    })
    .from(friendships)
    .where(sql`(${friendships.userA} = ${me} or ${friendships.userB} = ${me}) and ${friendships.status} in ('pending', 'accepted')`);
  const otherIds = rows.map((r) => (r.userA === me ? r.userB : r.userA));
  const found = otherIds.length ? await db.select(publicUser).from(users).where(inArray(users.id, otherIds)) : [];
  const byId = new Map(found.map((u) => [u.id, u]));
  const state: FriendsState = { friends: [], incoming: [], outgoing: [] };
  for (const r of rows) {
    const u = byId.get(r.userA === me ? r.userB : r.userA);
    if (!u) continue;
    if (r.status === "accepted") state.friends.push(u);
    else if (r.requestedBy === me) state.outgoing.push(u);
    else state.incoming.push(u);
  }
  return state;
}

// --- shared bills ------------------------------------------------------------------------------------

const splitNames = (data: Pick<BillData, "split">): string[] => (data.split ?? []).map((s) => s.name.trim().toLowerCase());

/**
 * Keeps who a bill is shared with in step with who is on it, after the owner saves it. A friend is added when
 * the bill names a contact linked to them (accepted friends only), but only if that name is NEW to the bill:
 * a past bill the owner chose not to share doesn't come back each time it is re-saved. A friend who is no
 * longer on the bill (or no longer a friend) loses their share.
 */
export async function syncShares(db: Db, ownerId: string, billId: string, data: Pick<BillData, "split">, previousNames: string[] | null): Promise<void> {
  const names = splitNames(data);
  const matched = names.length
    ? await db
        .select({ userId: people.linkedUserId, name: people.name, nameKey: people.nameKey })
        .from(people)
        .innerJoin(friendships, and(eq(friendships.status, "accepted"), sql`${friendships.userA} = least(${people.userId}, ${people.linkedUserId}) and ${friendships.userB} = greatest(${people.userId}, ${people.linkedUserId})`))
        .where(and(eq(people.userId, ownerId), sql`${people.linkedUserId} is not null`, inArray(people.nameKey, names)))
    : [];
  const existing = await db.select().from(billShares).where(eq(billShares.billId, billId));
  const have = new Map(existing.map((s) => [s.userId, s]));
  const matchedIds = new Set(matched.map((m) => m.userId as string));
  const before = new Set(previousNames ?? []);

  const add = matched.filter((m) => !have.has(m.userId as string) && !before.has(m.nameKey as string));
  if (add.length) {
    await db.insert(billShares).values(add.map((m) => ({ billId, userId: m.userId as string, asName: m.name }))).onConflictDoNothing();
  }
  for (const m of matched) {
    const cur = have.get(m.userId as string);
    if (cur && cur.asName !== m.name) await db.update(billShares).set({ asName: m.name }).where(and(eq(billShares.billId, billId), eq(billShares.userId, m.userId as string)));
  }
  const gone = existing.filter((s) => !matchedIds.has(s.userId)).map((s) => s.userId);
  if (gone.length) await db.delete(billShares).where(and(eq(billShares.billId, billId), inArray(billShares.userId, gone)));
}

export interface SharedBillListItem {
  id: string;
  hidden: boolean;
  billTitle: string;
  merchantCategory: string | null;
  totalAmount: number;
  currency: string;
  createdAt: Date;
  split: SplitRecord[] | null;
  settlement: Settlement | null;
  participants: Participant[] | null;
  asName: string;
  owner: PublicUser;
}

const sharedColumns = {
  id: billHistory.id,
  billTitle: billHistory.billTitle,
  merchantCategory: billHistory.merchantCategory,
  totalAmount: billHistory.totalAmount,
  currency: billHistory.currency,
  createdAt: billHistory.createdAt,
  split: sql<SplitRecord[] | null>`${billHistory.data} -> 'split'`,
  settlement: sql<Settlement | null>`${billHistory.data} -> 'settlement'`,
  participants: sql<Participant[] | null>`${billHistory.data} -> 'participants'`,
  asName: billShares.asName,
  hidden: billShares.hidden,
  ownerId: users.id,
  ownerUsername: users.username,
  ownerName: users.name,
  ownerAvatarSeed: users.avatarSeed,
};

type SharedRow = {
  id: string; hidden: boolean; billTitle: string; merchantCategory: string | null; totalAmount: number; currency: string; createdAt: Date;
  split: SplitRecord[] | null; settlement: Settlement | null; participants: Participant[] | null; asName: string;
  ownerId: string; ownerUsername: string | null; ownerName: string | null; ownerAvatarSeed: string | null;
};

const toShared = (r: SharedRow): SharedBillListItem => ({
  id: r.id, hidden: r.hidden, billTitle: r.billTitle, merchantCategory: r.merchantCategory, totalAmount: r.totalAmount, currency: r.currency, createdAt: r.createdAt,
  split: r.split, settlement: r.settlement, participants: r.participants, asName: r.asName,
  owner: { id: r.ownerId, username: r.ownerUsername, name: r.ownerName, avatarSeed: r.ownerAvatarSeed },
});

/** Bills friends have shared with me that I haven't hidden, newest first. Only while the friendship stands. */
export async function listSharedBills(db: Db, viewerId: string, opts: { withHidden?: boolean } = {}): Promise<SharedBillListItem[]> {
  const rows = await db
    .select(sharedColumns)
    .from(billShares)
    .innerJoin(billHistory, eq(billHistory.id, billShares.billId))
    .innerJoin(users, eq(users.id, billHistory.userId))
    .where(and(eq(billShares.userId, viewerId), opts.withHidden ? undefined : eq(billShares.hidden, false)))
    .orderBy(desc(billHistory.createdAt));
  return rows.map(toShared);
}

export interface SharedBillDetail extends SharedBillListItem {
  data: BillData;
  updatedAt: Date;
}

/**
 * ONE shared bill, read-only, for the friend it was shared with. A separate function (not getBill widened) and
 * it returns no client_id, so what a friend sees can never be saved back as one of their own bills.
 */
export async function getSharedBill(db: Db, viewerId: string, billId: string): Promise<SharedBillDetail | null> {
  const [row] = await db
    .select({ ...sharedColumns, data: billHistory.data, updatedAt: billHistory.updatedAt })
    .from(billShares)
    .innerJoin(billHistory, eq(billHistory.id, billShares.billId))
    .innerJoin(users, eq(users.id, billHistory.userId))
    .where(and(eq(billShares.userId, viewerId), eq(billShares.billId, billId)))
    .limit(1);
  if (!row) return null;
  return { ...toShared(row), data: row.data, updatedAt: row.updatedAt };
}

/** Hide (or show again) a shared bill in MY history. False when it isn't shared with me. */
export async function setShareHidden(db: Db, viewerId: string, billId: string, hidden: boolean): Promise<boolean> {
  const rows = await db.update(billShares).set({ hidden }).where(and(eq(billShares.userId, viewerId), eq(billShares.billId, billId))).returning({ id: billShares.billId });
  return rows.length > 0;
}

export interface PastBill {
  id: string;
  billTitle: string;
  totalAmount: number;
  currency: string;
  createdAt: Date;
}

/** My bills that name this linked friend's contact and aren't shared with them yet. */
export async function listSharablePastBills(db: Db, ownerId: string, friendId: string): Promise<PastBill[]> {
  if (!(await areFriends(db, ownerId, friendId))) return [];
  const [contact] = await db.select({ nameKey: people.nameKey }).from(people).where(and(eq(people.userId, ownerId), eq(people.linkedUserId, friendId))).limit(1);
  if (!contact?.nameKey) return [];
  return db
    .select({ id: billHistory.id, billTitle: billHistory.billTitle, totalAmount: billHistory.totalAmount, currency: billHistory.currency, createdAt: billHistory.createdAt })
    .from(billHistory)
    .where(
      and(
        eq(billHistory.userId, ownerId),
        sql`exists (select 1 from jsonb_array_elements(coalesce(${billHistory.data} -> 'split', '[]'::jsonb)) e where lower(btrim(e ->> 'name')) = ${contact.nameKey})`,
        sql`not exists (select 1 from ${billShares} s where s.bill_id = ${billHistory.id} and s.user_id = ${friendId})`,
      ),
    )
    .orderBy(desc(billHistory.createdAt))
    .limit(100);
}

/** Shares the chosen past bills (each must be mine and name that friend's contact) with the friend. */
export async function sharePastBills(db: Db, ownerId: string, friendId: string, billIds: string[]): Promise<number> {
  if (billIds.length === 0) return 0;
  const allowed = await listSharablePastBills(db, ownerId, friendId);
  const ok = allowed.filter((b) => billIds.includes(b.id));
  if (ok.length === 0) return 0;
  const [contact] = await db.select({ name: people.name }).from(people).where(and(eq(people.userId, ownerId), eq(people.linkedUserId, friendId))).limit(1);
  if (!contact) return 0;
  await db.insert(billShares).values(ok.map((b) => ({ billId: b.id, userId: friendId, asName: contact.name }))).onConflictDoNothing();
  return ok.length;
}

// --- payment claims ------------------------------------------------------------------------------------

export type ClaimResult =
  | { ok: true; claim: typeof paymentClaims.$inferSelect }
  | { ok: false; reason: "not-shared" | "no-such-transfer" | "already-paid" };

/**
 * A friend says they've paid one transfer on a shared bill. Re-checked against the STORED bill: the transfer
 * must exist, run from this friend (as the owner names them) to the owner, and still be unpaid. The amount is
 * the server's, never the client's. Claiming again after a rejection reopens it.
 */
export async function createClaim(db: Db, viewerId: string, billId: string, transferKey: string): Promise<ClaimResult> {
  const bill = await getSharedBill(db, viewerId, billId);
  if (!bill) return { ok: false, reason: "not-shared" };
  const view = settleView(bill.data.split ?? [], bill.data.settlement, localDate());
  const me = bill.asName.trim().toLowerCase();
  const ownerMe = (billMe(bill.data) ?? "").trim().toLowerCase();
  const t = view && view.ok ? view.transfers.find((x) => x.key === transferKey && x.from.trim().toLowerCase() === me && x.to.trim().toLowerCase() === ownerMe) : undefined;
  if (!t) return { ok: false, reason: "no-such-transfer" };
  if (t.paid) return { ok: false, reason: "already-paid" };
  const [claim] = await db
    .insert(paymentClaims)
    .values({ billId, userId: viewerId, transferKey: t.key, amount: t.amount })
    .onConflictDoUpdate({
      target: [paymentClaims.billId, paymentClaims.userId, paymentClaims.transferKey],
      set: { status: "pending", amount: t.amount, createdAt: sql`now()`, resolvedAt: null },
    })
    .returning();
  return { ok: true, claim };
}

/** My own claims on one shared bill. */
export const listMyClaims = (db: Db, viewerId: string, billId: string) =>
  db.select().from(paymentClaims).where(and(eq(paymentClaims.userId, viewerId), eq(paymentClaims.billId, billId)));

export interface OwnerClaim {
  id: string;
  billId: string;
  transferKey: string;
  amount: number;
  createdAt: Date;
  from: PublicUser;
  asName: string;
}

/** Claims waiting on ME for one of MY bills (stale ones, whose amount or transfer has changed, are left out). */
export async function listPendingClaimsForOwner(db: Db, ownerId: string, billId?: string): Promise<OwnerClaim[]> {
  const rows = await db
    .select({
      id: paymentClaims.id,
      billId: paymentClaims.billId,
      transferKey: paymentClaims.transferKey,
      amount: paymentClaims.amount,
      createdAt: paymentClaims.createdAt,
      asName: billShares.asName,
      split: sql<SplitRecord[] | null>`${billHistory.data} -> 'split'`,
      settlement: sql<Settlement | null>`${billHistory.data} -> 'settlement'`,
      fromId: users.id,
      fromUsername: users.username,
      fromName: users.name,
      fromAvatarSeed: users.avatarSeed,
    })
    .from(paymentClaims)
    .innerJoin(billShares, and(eq(billShares.billId, paymentClaims.billId), eq(billShares.userId, paymentClaims.userId)))
    .innerJoin(billHistory, eq(billHistory.id, paymentClaims.billId))
    .innerJoin(users, eq(users.id, paymentClaims.userId))
    .where(and(eq(billHistory.userId, ownerId), eq(paymentClaims.status, "pending"), billId ? eq(paymentClaims.billId, billId) : undefined))
    .orderBy(desc(paymentClaims.createdAt));
  const today = localDate();
  return rows
    .filter((r) => {
      const view = settleView(r.split ?? [], r.settlement ?? undefined, today);
      const t = view && view.ok ? view.transfers.find((x) => x.key === r.transferKey) : undefined;
      return !!t && !t.paid && t.amount === r.amount;
    })
    .map((r) => ({
      id: r.id, billId: r.billId, transferKey: r.transferKey, amount: r.amount, createdAt: r.createdAt, asName: r.asName,
      from: { id: r.fromId, username: r.fromUsername, name: r.fromName, avatarSeed: r.fromAvatarSeed },
    }));
}

/** Owner says "not received". False when the claim isn't on one of my bills. */
export async function rejectClaim(db: Db, ownerId: string, claimId: string): Promise<boolean> {
  const rows = await db
    .update(paymentClaims)
    .set({ status: "rejected", resolvedAt: sql`now()` })
    .where(
      and(
        eq(paymentClaims.id, claimId),
        eq(paymentClaims.status, "pending"),
        inArray(paymentClaims.billId, db.select({ id: billHistory.id }).from(billHistory).where(eq(billHistory.userId, ownerId))),
      ),
    )
    .returning({ id: paymentClaims.id });
  return rows.length > 0;
}

/** After the owner saves a bill: a pending claim whose paid mark now exists (same amount) is confirmed. */
export async function confirmSatisfiedClaims(db: Db, billId: string, settlement: Settlement | undefined): Promise<void> {
  const pending = await db.select().from(paymentClaims).where(and(eq(paymentClaims.billId, billId), eq(paymentClaims.status, "pending")));
  const done = pending.filter((c) => settlement?.paid?.[c.transferKey]?.amount === c.amount).map((c) => c.id);
  if (done.length) await db.update(paymentClaims).set({ status: "confirmed", resolvedAt: sql`now()` }).where(inArray(paymentClaims.id, done));
}
