import { and, desc, eq, inArray, isNull, sql } from "drizzle-orm";

import { ownerLabel } from "@/lib/split/shared";
import { notificationHref, notificationText } from "@/lib/notifications/text";
import { afterResponse } from "@/lib/push/after";
import { pushConfigured, pushToUser } from "@/lib/push/send";
import type { Db } from "./client";
import { notifications, users, type NotificationKind } from "./schema";

// The inbox. Same rule as the rest of lib/db: every read and write is scoped by the session's userId, so one
// user can never list, count or mark another user's notifications.

export interface NotifyInput {
  userId: string;
  kind: NotificationKind;
  actorId?: string | null;
  billId?: string | null;
  payload?: Record<string, string | number>;
  /** Same key for the same event: a retry (or a second save) can't notify twice. */
  dedupe: string;
  /** Bring an existing one back as new (unread, newest) instead of ignoring the repeat. */
  refresh?: boolean;
}

/** Records an event. Returns true when it is new (or refreshed), false when it was a repeat. Never notifies yourself. */
export async function notify(db: Db, n: NotifyInput): Promise<boolean> {
  if (n.actorId && n.actorId === n.userId) return false;
  const values = { userId: n.userId, kind: n.kind, actorId: n.actorId ?? null, billId: n.billId ?? null, payload: n.payload ?? {}, dedupe: n.dedupe };
  const q = db.insert(notifications).values(values);
  const rows = n.refresh
    ? await q
        .onConflictDoUpdate({
          target: [notifications.userId, notifications.dedupe],
          set: { kind: values.kind, actorId: values.actorId, billId: values.billId, payload: values.payload, readAt: null, createdAt: sql`now()` },
        })
        .returning({ id: notifications.id })
    : await q.onConflictDoNothing().returning({ id: notifications.id });
  if (rows.length === 0) return false;
  if (pushConfigured()) afterResponse(() => pushNew(db, n));
  return true;
}

/** The phone side of a new notification: the same sentence the inbox shows, to every device the user turned on. */
async function pushNew(db: Db, n: NotifyInput): Promise<void> {
  let actor = "Someone";
  if (n.actorId) {
    const [u] = await db.select({ name: users.name, username: users.username }).from(users).where(eq(users.id, n.actorId)).limit(1);
    if (u) actor = ownerLabel(u);
  }
  const like = { kind: n.kind, billId: n.billId ?? null, payload: n.payload ?? {}, actorId: n.actorId ?? null };
  await pushToUser(db, n.userId, { title: "Bill.a", body: notificationText(like, actor), url: notificationHref(like) ?? "/dashboard/notifications", tag: n.kind });
}

/** The one-time hello for a new account: points at the optional AI key guide. Once per account (the dedupe key). */
export async function welcomeNewUser(db: Db, userId: string): Promise<void> {
  await notify(db, { userId, kind: "welcome", dedupe: "welcome" });
}

export async function notifyMany(db: Db, list: NotifyInput[]): Promise<void> {
  for (const n of list) await notify(db, n);
}

export interface InboxItem {
  id: string;
  kind: NotificationKind;
  billId: string | null;
  payload: Record<string, string | number>;
  readAt: Date | null;
  createdAt: Date;
  actorId: string | null;
  actor: { name: string | null; username: string | null; avatarSeed: string | null } | null;
}

export async function listNotifications(db: Db, userId: string, limit = 50): Promise<InboxItem[]> {
  const rows = await db
    .select({
      id: notifications.id,
      kind: notifications.kind,
      billId: notifications.billId,
      payload: notifications.payload,
      readAt: notifications.readAt,
      createdAt: notifications.createdAt,
      actorId: notifications.actorId,
      actorName: users.name,
      actorUsername: users.username,
      actorAvatarSeed: users.avatarSeed,
    })
    .from(notifications)
    .leftJoin(users, eq(users.id, notifications.actorId))
    .where(eq(notifications.userId, userId))
    .orderBy(desc(notifications.createdAt))
    .limit(limit);
  return rows.map((r) => ({
    id: r.id,
    kind: r.kind,
    billId: r.billId,
    payload: r.payload,
    readAt: r.readAt,
    createdAt: r.createdAt,
    actorId: r.actorId,
    actor: r.actorId ? { name: r.actorName, username: r.actorUsername, avatarSeed: r.actorAvatarSeed } : null,
  }));
}

export async function unreadCount(db: Db, userId: string): Promise<number> {
  const [row] = await db
    .select({ n: sql<number>`count(*)::int` })
    .from(notifications)
    .where(and(eq(notifications.userId, userId), isNull(notifications.readAt)));
  return row?.n ?? 0;
}

/** Marks some of MY notifications read. Ids that are someone else's are ignored. Returns how many changed. */
export async function markRead(db: Db, userId: string, ids: string[]): Promise<number> {
  if (ids.length === 0) return 0;
  const rows = await db
    .update(notifications)
    .set({ readAt: sql`now()` })
    .where(and(eq(notifications.userId, userId), inArray(notifications.id, ids), isNull(notifications.readAt)))
    .returning({ id: notifications.id });
  return rows.length;
}

export async function markAllRead(db: Db, userId: string): Promise<number> {
  const rows = await db
    .update(notifications)
    .set({ readAt: sql`now()` })
    .where(and(eq(notifications.userId, userId), isNull(notifications.readAt)))
    .returning({ id: notifications.id });
  return rows.length;
}
