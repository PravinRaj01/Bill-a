"use server";

import { z } from "zod";
import { getUserIdOrNull, requireUserId } from "@/lib/auth/session";
import { getDb } from "@/lib/db/client";
import * as n from "@/lib/db/notifications";
import * as s from "@/lib/db/social";
import { ownerLabel } from "@/lib/split/shared";
import { notificationHref, notificationText } from "@/lib/notifications/text";
import type { NotificationKind } from "@/lib/db/schema";

// The inbox. The user comes from the session only; ids from the browser are only ever matched against that
// user's own rows.

export interface InboxEntry {
  id: string;
  kind: NotificationKind;
  text: string;
  href: string | null;
  read: boolean;
  createdAt: string;
  actor: { name: string; avatarSeed: string | null } | null;
}

export async function getInbox(): Promise<InboxEntry[]> {
  const userId = await getUserIdOrNull();
  if (!userId) return [];
  const rows = await n.listNotifications(getDb(), userId);
  return rows.map((r) => {
    const name = r.actor ? ownerLabel(r.actor) : "Someone";
    return {
      id: r.id,
      kind: r.kind,
      text: notificationText(r, name),
      href: notificationHref(r),
      read: !!r.readAt,
      createdAt: r.createdAt.toISOString(),
      actor: r.actor ? { name, avatarSeed: r.actor.avatarSeed } : null,
    };
  });
}

export async function getUnreadCount(): Promise<number> {
  const userId = await getUserIdOrNull();
  return userId ? n.unreadCount(getDb(), userId) : 0;
}

export async function markNotificationsRead(ids: string[]): Promise<number> {
  const userId = await requireUserId();
  return n.markRead(getDb(), userId, z.array(z.uuid()).max(100).parse(ids));
}

export async function markAllNotificationsRead(): Promise<number> {
  const userId = await requireUserId();
  return n.markAllRead(getDb(), userId);
}

/** Clear some notifications from the inbox. Returns the ids actually cleared (to Undo). */
export async function dismissNotifications(ids: string[]): Promise<string[]> {
  const userId = await requireUserId();
  return n.dismiss(getDb(), userId, z.array(z.uuid()).max(100).parse(ids));
}

/** Clear the whole inbox. Returns the ids cleared (to Undo). */
export async function clearAllNotifications(): Promise<string[]> {
  const userId = await requireUserId();
  return n.dismissAll(getDb(), userId);
}

export async function restoreNotifications(ids: string[]): Promise<number> {
  const userId = await requireUserId();
  return n.restore(getDb(), userId, z.array(z.uuid()).max(500).parse(ids));
}

export interface NudgeTarget {
  friendId: string;
  name: string;
  owed: number;
  nudgedToday: boolean;
}

/** Friends on one of my shared bills who still owe me, for the "remind in the app" buttons. */
export async function nudgeTargets(billId: string): Promise<NudgeTarget[]> {
  const userId = await getUserIdOrNull();
  if (!userId || !z.uuid().safeParse(billId).success) return [];
  const rows = await s.listNudgeable(getDb(), userId, billId);
  return rows.map((r) => ({ friendId: r.friend.id, name: ownerLabel(r.friend), owed: r.owed, nudgedToday: r.nudgedToday }));
}

export type NudgeOutcome = { ok: true } | { ok: false; error: string };

export async function nudgeFriend(billId: string, friendId: string): Promise<NudgeOutcome> {
  const userId = await requireUserId();
  const r = await s.nudgeSharedFriend(getDb(), userId, z.uuid().parse(billId), z.string().min(1).max(64).parse(friendId));
  if (r.ok) return r;
  const messages = {
    "not-shared": "That bill isn't shared with them.",
    "nothing-owed": "They don't owe you anything on this bill.",
    "already-today": "You've already reminded them today.",
  } as const;
  return { ok: false, error: messages[r.reason] };
}
