import { and, desc, eq, inArray, or, sql } from "drizzle-orm";

import { billMe } from "@/lib/split/participants";
import { dateInZone } from "@/lib/notifications/digest";
import { settleView, localDate } from "@/lib/split/settle";
import type { BillData, Participant, Settlement, SplitRecord } from "@/types/domain";
import type { Db } from "./client";
import { notify } from "./notifications";
import { billHistory, billShares, friendships, notifications, paymentClaims, people, users } from "./schema";

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
    await notify(db, { userId: otherId, kind: "friend_request", actorId: me, dedupe: `friend_request:${me}`, refresh: true });
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
  await notify(db, { userId: otherId, kind: "friend_request", actorId: me, dedupe: `friend_request:${me}`, refresh: true });
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
    await notify(db, { userId: otherId, kind: "friend_accepted", actorId: me, dedupe: `friend_accepted:${me}`, refresh: true });
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

/** Which of these accounts are accepted friends of the owner. */
export async function acceptedFriendIds(db: Db, ownerId: string, ids: string[]): Promise<Set<string>> {
  const unique = [...new Set(ids.filter((i) => i && i !== ownerId))];
  if (unique.length === 0) return new Set();
  const rows = await db
    .select({ a: friendships.userA, b: friendships.userB })
    .from(friendships)
    .where(and(eq(friendships.status, "accepted"), or(and(eq(friendships.userA, ownerId), inArray(friendships.userB, unique)), and(eq(friendships.userB, ownerId), inArray(friendships.userA, unique)))));
  return new Set(rows.map((r) => (r.a === ownerId ? r.b : r.a)));
}

/**
 * A bill as it will be stored: a participant's account link is kept only when that account is an accepted friend
 * of the owner (and is not the owner). The client says who a name stands for, the server decides if that is allowed,
 * so a forged id creates no share and leaves nothing behind in the stored bill.
 */
export async function withFriendIds<T extends Pick<BillData, "participants">>(db: Db, ownerId: string, data: T): Promise<T> {
  const parts = data.participants;
  if (!parts || !parts.some((p) => p.userId)) return data;
  const ok = await acceptedFriendIds(db, ownerId, parts.flatMap((p) => (p.userId ? [p.userId] : [])));
  return { ...data, participants: parts.map((p) => (p.userId && !p.me && ok.has(p.userId) ? p : { name: p.name, ...(p.me ? { me: true as const } : {}) })) };
}

/**
 * Shares follow the accounts a bill's participants are linked to: a linked friend gets the bill, one who is no longer
 * linked loses it. The name only says what the owner calls them (`asName`), so it can change without touching the share.
 */
async function syncSharesById(db: Db, ownerId: string, billId: string, participants: Participant[]): Promise<void> {
  const wanted = new Map<string, string>();
  for (const p of participants) if (p.userId && !p.me && p.userId !== ownerId) wanted.set(p.userId, p.name);
  const existing = await db.select().from(billShares).where(eq(billShares.billId, billId));
  const have = new Map(existing.map((s) => [s.userId, s]));

  const add = [...wanted].filter(([id]) => !have.has(id));
  if (add.length) {
    await db.insert(billShares).values(add.map(([userId, asName]) => ({ billId, userId, asName }))).onConflictDoNothing();
    const [bill] = await db.select({ title: billHistory.billTitle }).from(billHistory).where(eq(billHistory.id, billId)).limit(1);
    // refresh: a friend added again after being taken off the bill hears about it again.
    for (const [userId] of add) await notify(db, { userId, kind: "bill_added", actorId: ownerId, billId, payload: { title: bill?.title ?? "a bill" }, dedupe: `bill_added:${billId}`, refresh: true });
  }
  for (const [id, name] of wanted) {
    const cur = have.get(id);
    if (cur && cur.asName !== name) await db.update(billShares).set({ asName: name }).where(and(eq(billShares.billId, billId), eq(billShares.userId, id)));
  }
  const gone = existing.filter((s) => !wanted.has(s.userId)).map((s) => s.userId);
  if (gone.length) {
    await db.delete(billShares).where(and(eq(billShares.billId, billId), inArray(billShares.userId, gone)));
    const [bill] = await db.select({ title: billHistory.billTitle }).from(billHistory).where(eq(billHistory.id, billId)).limit(1);
    // The bill is still there, but it has left their History: say so, once per time it happens.
    for (const userId of gone) await notify(db, { userId, kind: "bill_unshared", actorId: ownerId, billId, payload: { title: bill?.title ?? "a bill" }, dedupe: `bill_unshared:${billId}`, refresh: true });
  }
}

/**
 * Keeps who a bill is shared with in step with who is on it, after the owner saves it. A bill that records its
 * participants is shared by account (syncSharesById); an older one, saved before that, keeps matching by name.
 */
export async function syncShares(db: Db, ownerId: string, billId: string, data: Pick<BillData, "split" | "participants">, previousNames: string[] | null): Promise<void> {
  if (data.participants) return syncSharesById(db, ownerId, billId, data.participants);
  return syncSharesByName(db, ownerId, billId, data, previousNames);
}

/**
 * (Older bills, no participants.) Keeps who a bill is shared with in step with who is on it. A friend is added when
 * the bill names a contact linked to them (accepted friends only), but only if that name is NEW to the bill:
 * a past bill the owner chose not to share doesn't come back each time it is re-saved. A friend who is no
 * longer on the bill (or no longer a friend) loses their share.
 */
async function syncSharesByName(db: Db, ownerId: string, billId: string, data: Pick<BillData, "split">, previousNames: string[] | null): Promise<void> {
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
    const [bill] = await db.select({ title: billHistory.billTitle }).from(billHistory).where(eq(billHistory.id, billId)).limit(1);
    for (const m of add) await notify(db, { userId: m.userId as string, kind: "bill_added", actorId: ownerId, billId, payload: { title: bill?.title ?? "a bill" }, dedupe: `bill_added:${billId}` });
  }
  for (const m of matched) {
    const cur = have.get(m.userId as string);
    if (cur && cur.asName !== m.name) await db.update(billShares).set({ asName: m.name }).where(and(eq(billShares.billId, billId), eq(billShares.userId, m.userId as string)));
  }
  // A share also stays while the bill still names the friend as it was shared (a past bill shared under their
  // first name, say, which isn't their contact's name).
  const gone = existing.filter((s) => !matchedIds.has(s.userId) && !names.includes(s.asName.trim().toLowerCase())).map((s) => s.userId);
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
  /** How the friend is written on this bill (the name the share points at). */
  asName: string;
}

/**
 * My bills that name this linked friend and aren't shared with them yet. The friend may be written as their
 * contact's name, their first name or their @username (the contact is created from the full account name, while
 * bills usually say the first name). The bill's own "me" never counts, in case we share a first name. The owner
 * confirms each bill before it is shared, so a looser match only ever offers more.
 */
export async function listSharablePastBills(db: Db, ownerId: string, friendId: string): Promise<PastBill[]> {
  if (!(await areFriends(db, ownerId, friendId))) return [];
  const [contact] = await db.select({ nameKey: people.nameKey }).from(people).where(and(eq(people.userId, ownerId), eq(people.linkedUserId, friendId))).limit(1);
  if (!contact?.nameKey) return [];
  const [friend] = await db.select({ name: users.name, username: users.username }).from(users).where(eq(users.id, friendId)).limit(1);
  const firstName = (friend?.name ?? "").trim().split(/\s+/)[0]?.toLowerCase();
  // Their first name also counts, but only when no other contact of mine goes by it (two Pravins: no guessing).
  const others = (await db.select({ nameKey: people.nameKey }).from(people).where(eq(people.userId, ownerId))).filter((o) => o.nameKey !== contact.nameKey);
  const firstIsFree = !!firstName && !others.some((o) => o.nameKey === firstName || o.nameKey?.startsWith(`${firstName} `));
  const keys = [...new Set([contact.nameKey, firstIsFree ? firstName : null].filter((k): k is string => !!k))];
  const keyList = sql.join(keys.map((k) => sql`${k}`), sql`, `);
  const entries = sql`jsonb_array_elements(coalesce(${billHistory.data} -> 'split', '[]'::jsonb))`;
  // The bill's own "me" never counts (a shared first name can't match yourself): from its participants, else the old guess.
  const owner = sql`lower(btrim(coalesce((select p ->> 'name' from jsonb_array_elements(coalesce(${billHistory.data} -> 'participants', '[]'::jsonb)) p where p ->> 'me' = 'true' limit 1), ${billHistory.data} -> 'settlement' ->> 'me', '')))`;
  const named = sql`lower(btrim(e ->> 'name')) in (${keyList}) and lower(btrim(e ->> 'name')) <> ${owner}`;
  const rows = await db
    .select({
      id: billHistory.id,
      billTitle: billHistory.billTitle,
      totalAmount: billHistory.totalAmount,
      currency: billHistory.currency,
      createdAt: billHistory.createdAt,
      // The contact's own name wins when a bill has more than one candidate.
      asName: sql<string>`(select e ->> 'name' from ${entries} e where ${named} order by (lower(btrim(e ->> 'name')) = ${contact.nameKey}) desc limit 1)`,
    })
    .from(billHistory)
    .where(and(eq(billHistory.userId, ownerId), sql`exists (select 1 from ${entries} e where ${named})`, sql`not exists (select 1 from ${billShares} s where s.bill_id = ${billHistory.id} and s.user_id = ${friendId})`))
    .orderBy(desc(billHistory.createdAt))
    .limit(100);
  return rows.map((r) => ({ ...r, asName: r.asName.trim() }));
}

export interface LinkCandidate {
  id: string;
  name: string;
  phone: string | null;
}

/** Contacts of mine, not yet linked, that could be this friend: named like their full name, first name or @username. */
export async function linkCandidates(db: Db, ownerId: string, friendId: string): Promise<LinkCandidate[]> {
  if (!(await areFriends(db, ownerId, friendId))) return [];
  const [friend] = await db.select({ name: users.name, username: users.username }).from(users).where(eq(users.id, friendId)).limit(1);
  const full = (friend?.name ?? "").trim().toLowerCase();
  const first = full.split(" ")[0];
  const keys = [...new Set([full, first, friend?.username?.toLowerCase()].filter((k): k is string => !!k))];
  if (keys.length === 0) return [];
  return db
    .select({ id: people.id, name: people.name, phone: people.phone })
    .from(people)
    .where(and(eq(people.userId, ownerId), sql`${people.linkedUserId} is null`, inArray(people.nameKey, keys)))
    .orderBy(people.nameKey);
}

/**
 * "This contact is my friend": links an existing contact (keeping its name and number) to the friend and drops the
 * contact that was made for them automatically. Only for a friend, and only an unlinked contact of mine.
 */
export async function linkContactToFriend(db: Db, ownerId: string, friendId: string, contactId: string): Promise<boolean> {
  if (!(await areFriends(db, ownerId, friendId))) return false;
  const [chosen] = await db.select({ id: people.id, phone: people.phone }).from(people).where(and(eq(people.id, contactId), eq(people.userId, ownerId), sql`${people.linkedUserId} is null`)).limit(1);
  if (!chosen) return false;
  const [auto] = await db.select({ id: people.id, phone: people.phone }).from(people).where(and(eq(people.userId, ownerId), eq(people.linkedUserId, friendId))).limit(1);
  // The partial unique index allows one linked contact per friend, so the automatic one goes first (its number moves over).
  if (auto) await db.delete(people).where(and(eq(people.id, auto.id), eq(people.userId, ownerId)));
  await db.update(people).set({ linkedUserId: friendId, ...(chosen.phone || !auto?.phone ? {} : { phone: auto.phone }), updatedAt: sql`now()` }).where(and(eq(people.id, chosen.id), eq(people.userId, ownerId)));
  return true;
}

/** Shares the chosen past bills (each must be mine and name that friend) with the friend. */
export async function sharePastBills(db: Db, ownerId: string, friendId: string, billIds: string[]): Promise<number> {
  if (billIds.length === 0) return 0;
  const allowed = await listSharablePastBills(db, ownerId, friendId);
  const ok = allowed.filter((b) => billIds.includes(b.id));
  if (ok.length === 0) return 0;
  await db.insert(billShares).values(ok.map((b) => ({ billId: b.id, userId: friendId, asName: b.asName }))).onConflictDoNothing();
  // A bill that records its participants links the friend there too, so the share follows the account, not the name.
  const rows = await db.select({ id: billHistory.id, data: billHistory.data }).from(billHistory).where(and(eq(billHistory.userId, ownerId), inArray(billHistory.id, ok.map((b) => b.id))));
  for (const r of rows) {
    const asName = ok.find((b) => b.id === r.id)?.asName.trim().toLowerCase();
    const parts = r.data.participants;
    if (!parts || !asName) continue;
    const linked = parts.map((p) => (!p.me && p.name.trim().toLowerCase() === asName ? { ...p, userId: friendId } : p));
    await db.update(billHistory).set({ data: { ...r.data, participants: linked } }).where(eq(billHistory.id, r.id));
  }
  for (const b of ok) await notify(db, { userId: friendId, kind: "bill_added", actorId: ownerId, billId: b.id, payload: { title: b.billTitle }, dedupe: `bill_added:${b.id}` });
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
  await notify(db, {
    userId: bill.owner.id,
    kind: "claim_made",
    actorId: viewerId,
    billId,
    payload: { title: bill.billTitle, amount: t.amount, currency: bill.currency },
    dedupe: `claim_made:${claim.id}`,
    refresh: true,
  });
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
    .returning({ id: paymentClaims.id, userId: paymentClaims.userId, billId: paymentClaims.billId, amount: paymentClaims.amount });
  for (const c of rows) {
    const [bill] = await db.select({ title: billHistory.billTitle, currency: billHistory.currency }).from(billHistory).where(eq(billHistory.id, c.billId)).limit(1);
    await notify(db, { userId: c.userId, kind: "claim_rejected", actorId: ownerId, billId: c.billId, payload: { title: bill?.title ?? "a bill", amount: c.amount, currency: bill?.currency ?? "" }, dedupe: `claim_rejected:${c.id}`, refresh: true });
  }
  return rows.length > 0;
}

/**
 * A claim whose transfer is gone, or whose amount no longer matches, can never be confirmed (the owner doesn't even
 * see it). Close it and tell the friend, so they aren't left on "waiting" and can claim again against the new bill.
 */
async function closeStaleClaims(db: Db, billId: string, pending: (typeof paymentClaims.$inferSelect)[], settlement: Settlement | undefined, split: SplitRecord[] | undefined): Promise<void> {
  if (!split || pending.length === 0) return;
  const view = settleView(split, settlement, localDate());
  if (!view || !view.ok) return;
  const stale = pending.filter((c) => {
    if (settlement?.paid?.[c.transferKey]?.amount === c.amount) return false; // about to be confirmed
    const t = view.transfers.find((x) => x.key === c.transferKey);
    return !t || t.paid || t.amount !== c.amount;
  });
  if (stale.length === 0) return;
  await db.update(paymentClaims).set({ status: "rejected", resolvedAt: sql`now()` }).where(inArray(paymentClaims.id, stale.map((c) => c.id)));
  const [bill] = await db.select({ ownerId: billHistory.userId, title: billHistory.billTitle }).from(billHistory).where(eq(billHistory.id, billId)).limit(1);
  for (const c of stale) await notify(db, { userId: c.userId, kind: "claim_changed", actorId: bill?.ownerId ?? null, billId, payload: { title: bill?.title ?? "a bill" }, dedupe: `claim_changed:${c.id}:${Date.now()}` });
}

/** After the owner saves a bill: a pending claim whose paid mark now exists (same amount) is confirmed. */
export async function confirmSatisfiedClaims(db: Db, billId: string, settlement: Settlement | undefined, split?: SplitRecord[]): Promise<void> {
  const pending = await db.select().from(paymentClaims).where(and(eq(paymentClaims.billId, billId), eq(paymentClaims.status, "pending")));
  await closeStaleClaims(db, billId, pending, settlement, split);
  const done = pending.filter((c) => settlement?.paid?.[c.transferKey]?.amount === c.amount).map((c) => c.id);
  if (done.length === 0) return;
  await db.update(paymentClaims).set({ status: "confirmed", resolvedAt: sql`now()` }).where(inArray(paymentClaims.id, done));
  const [bill] = await db.select({ ownerId: billHistory.userId, title: billHistory.billTitle, currency: billHistory.currency }).from(billHistory).where(eq(billHistory.id, billId)).limit(1);
  for (const c of pending.filter((x) => done.includes(x.id))) {
    await notify(db, { userId: c.userId, kind: "claim_confirmed", actorId: bill?.ownerId ?? null, billId, payload: { title: bill?.title ?? "a bill", amount: c.amount, currency: bill?.currency ?? "" }, dedupe: `claim_confirmed:${c.id}`, refresh: true });
  }
}

// --- nudges ----------------------------------------------------------------------------------------------

/** What this friend still owes the owner on the bill, from the stored data (0 when untracked or settled). */
function owedToOwner(data: BillData, asName: string): number {
  const view = settleView(data.split ?? [], data.settlement, localDate());
  const friend = asName.trim().toLowerCase();
  const owner = (billMe(data) ?? "").trim().toLowerCase();
  return view && view.ok ? view.transfers.filter((t) => !t.paid && t.from.trim().toLowerCase() === friend && t.to.trim().toLowerCase() === owner).reduce((n, t) => n + t.amount, 0) : 0;
}

export interface Nudgeable {
  friend: PublicUser;
  asName: string;
  owed: number;
  nudgedToday: boolean;
}

/** Friends on MY bill who still owe me, and whether I have already nudged them today. */
/** "Today" for the owner, in their own time zone (the digest already works this way). */
async function ownerDay(db: Db, ownerId: string): Promise<string> {
  const [u] = await db.select({ tz: users.tz }).from(users).where(eq(users.id, ownerId)).limit(1);
  return dateInZone(new Date(), u?.tz ?? "");
}

export async function listNudgeable(db: Db, ownerId: string, billId: string, day?: string): Promise<Nudgeable[]> {
  const today = day ?? (await ownerDay(db, ownerId));
  const rows = await db
    .select({ asName: billShares.asName, data: billHistory.data, ...publicUser })
    .from(billShares)
    .innerJoin(billHistory, eq(billHistory.id, billShares.billId))
    .innerJoin(users, eq(users.id, billShares.userId))
    .where(and(eq(billHistory.id, billId), eq(billHistory.userId, ownerId)));
  const out: Nudgeable[] = [];
  for (const r of rows) {
    const owed = owedToOwner(r.data, r.asName);
    if (owed <= 0) continue;
    const [sent] = await db.select({ id: notifications.id }).from(notifications).where(and(eq(notifications.userId, r.id), eq(notifications.dedupe, `nudge:${billId}:${today}`))).limit(1);
    out.push({ friend: { id: r.id, username: r.username, name: r.name, avatarSeed: r.avatarSeed }, asName: r.asName, owed, nudgedToday: !!sent });
  }
  return out;
}

export type NudgeResult = { ok: true } | { ok: false; reason: "not-shared" | "nothing-owed" | "already-today" };

/**
 * The owner of a shared bill nudges a friend who still owes them on it, in the friend's inbox. Once a day per
 * bill and friend. What they owe comes from the stored bill (never from the caller), and so does the text.
 */
export async function nudgeSharedFriend(db: Db, ownerId: string, billId: string, friendId: string, day?: string): Promise<NudgeResult> {
  const today = day ?? (await ownerDay(db, ownerId));
  const [row] = await db
    .select({ asName: billShares.asName, data: billHistory.data, title: billHistory.billTitle, currency: billHistory.currency })
    .from(billShares)
    .innerJoin(billHistory, eq(billHistory.id, billShares.billId))
    .where(and(eq(billHistory.id, billId), eq(billHistory.userId, ownerId), eq(billShares.userId, friendId)))
    .limit(1);
  if (!row) return { ok: false, reason: "not-shared" };
  const owed = owedToOwner(row.data, row.asName);
  if (owed <= 0) return { ok: false, reason: "nothing-owed" };
  const fresh = await notify(db, { userId: friendId, kind: "nudge", actorId: ownerId, billId, payload: { title: row.title, amount: owed, currency: row.currency }, dedupe: `nudge:${billId}:${today}` });
  return fresh ? { ok: true } : { ok: false, reason: "already-today" };
}
