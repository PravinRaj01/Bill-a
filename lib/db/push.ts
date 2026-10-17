import { and, eq } from "drizzle-orm";

import type { Db } from "./client";
import { pushSubscriptions, type PushSubscriptionRow } from "./schema";

// Which devices a user has said yes to push on. Same rule as the rest of lib/db: everything is scoped by the
// session's userId, so nobody can list or remove another user's devices.

export interface SubscriptionInput {
  endpoint: string;
  p256dh: string;
  auth: string;
}

/** Saves this device for me. If the device was someone else's (a shared phone), it moves to me. */
export async function saveSubscription(db: Db, userId: string, s: SubscriptionInput): Promise<void> {
  await db
    .insert(pushSubscriptions)
    .values({ endpoint: s.endpoint, userId, p256dh: s.p256dh, auth: s.auth })
    .onConflictDoUpdate({ target: pushSubscriptions.endpoint, set: { userId, p256dh: s.p256dh, auth: s.auth, createdAt: new Date() } });
}

/** Removes one of MY devices. False when that endpoint isn't mine (or isn't there). */
export async function removeSubscription(db: Db, userId: string, endpoint: string): Promise<boolean> {
  const rows = await db
    .delete(pushSubscriptions)
    .where(and(eq(pushSubscriptions.userId, userId), eq(pushSubscriptions.endpoint, endpoint)))
    .returning({ e: pushSubscriptions.endpoint });
  return rows.length > 0;
}

export const listSubscriptions = (db: Db, userId: string): Promise<PushSubscriptionRow[]> =>
  db.select().from(pushSubscriptions).where(eq(pushSubscriptions.userId, userId));

/** Drops a device the push service said is gone. Not scoped by user: it is the server's own cleanup. */
export async function dropDeadSubscription(db: Db, endpoint: string): Promise<void> {
  await db.delete(pushSubscriptions).where(eq(pushSubscriptions.endpoint, endpoint));
}
