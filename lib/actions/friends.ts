"use server";

import { z } from "zod";
import { getUserIdOrNull, requireUserId } from "@/lib/auth/session";
import { getDb } from "@/lib/db/client";
import * as s from "@/lib/db/social";

// Friends. The user comes from the session only. Searching is by exact @username (never partial), and a
// stranger's id is never sent to the browser: only people already related to you (friends, requests) carry one.

const handleSchema = z
  .string()
  .trim()
  .transform((h) => h.replace(/^@/, "").toLowerCase())
  .pipe(z.string().regex(/^[a-z0-9_]{3,20}$/));
const idSchema = z.string().min(1).max(64);

export interface FriendView {
  id: string;
  username: string | null;
  name: string | null;
  avatarSeed: string | null;
}

export interface FriendsView {
  friends: FriendView[];
  incoming: FriendView[];
  outgoing: FriendView[];
}

const empty: FriendsView = { friends: [], incoming: [], outgoing: [] };

export async function getFriends(): Promise<FriendsView> {
  const userId = await getUserIdOrNull();
  return userId ? s.listFriendsState(getDb(), userId) : empty;
}

export type LookupResult =
  | { ok: true; user: { username: string; name: string | null; avatarSeed: string | null }; relation: s.Relation | "self" }
  | { ok: false; error: string };

const NOT_FOUND = "No one has that username.";

/** Who is @handle? Name and portrait only, plus how they relate to you. The same answer for "no such user" always. */
export async function lookupUser(handle: string): Promise<LookupResult> {
  const userId = await requireUserId();
  const parsed = handleSchema.safeParse(handle);
  if (!parsed.success) return { ok: false, error: NOT_FOUND };
  const db = getDb();
  const user = await s.findUserByUsername(db, parsed.data);
  if (!user || !user.username) return { ok: false, error: NOT_FOUND };
  const relation = user.id === userId ? "self" : await s.relationTo(db, userId, user.id);
  return { ok: true, user: { username: user.username, name: user.name, avatarSeed: user.avatarSeed }, relation };
}

export type RequestOutcome = { ok: true; status: "requested" | "accepted" } | { ok: false; error: string };

export async function sendFriendRequest(handle: string): Promise<RequestOutcome> {
  const userId = await requireUserId();
  const parsed = handleSchema.safeParse(handle);
  if (!parsed.success) return { ok: false, error: NOT_FOUND };
  const db = getDb();
  const user = await s.findUserByUsername(db, parsed.data);
  if (!user) return { ok: false, error: NOT_FOUND };
  const r = await s.requestFriend(db, userId, user.id);
  if (r.ok) return r;
  const messages: Record<typeof r.reason, string> = {
    self: "That's you.",
    "not-found": NOT_FOUND,
    "already-friends": "You're already friends.",
    "already-requested": "You've already sent a request.",
    "too-many": "You have too many requests waiting. Try again once some are answered.",
  };
  return { ok: false, error: messages[r.reason] };
}

export async function respondToRequest(fromUserId: string, accept: boolean): Promise<boolean> {
  const userId = await requireUserId();
  return s.answerRequest(getDb(), userId, idSchema.parse(fromUserId), accept);
}

export async function unfriend(friendUserId: string): Promise<void> {
  const userId = await requireUserId();
  await s.removeFriend(getDb(), userId, idSchema.parse(friendUserId));
}

export interface PastBillView {
  id: string;
  billTitle: string;
  totalAmount: number;
  currency: string;
  createdAt: string;
}

/** My bills that name this friend and aren't shared with them yet (to offer sharing them). */
export async function pastBillsFor(friendUserId: string): Promise<PastBillView[]> {
  const userId = await getUserIdOrNull();
  if (!userId) return [];
  const rows = await s.listSharablePastBills(getDb(), userId, idSchema.parse(friendUserId));
  return rows.map((b) => ({ id: b.id, billTitle: b.billTitle, totalAmount: b.totalAmount, currency: b.currency, createdAt: b.createdAt.toISOString() }));
}

export async function sharePastBills(friendUserId: string, billIds: string[]): Promise<number> {
  const userId = await requireUserId();
  return s.sharePastBills(getDb(), userId, idSchema.parse(friendUserId), z.array(z.uuid()).max(100).parse(billIds));
}
