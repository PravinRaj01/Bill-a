import { and, eq, inArray, sql } from "drizzle-orm";

import type { Db } from "./client";
import { pushSubscriptions, users } from "./schema";

// Push devices. Every read and write is scoped by the session's userId; the endpoint is the browser's own
// address for the device, so a device that switches accounts is moved to the new one, never duplicated.

export interface PushKeys {
  endpoint: string;
  p256dh: string;
  auth: string;
}

export async function saveSubscription(db: Db, userId: string, s: PushKeys): Promise<void> {
  await db
    .insert(pushSubscriptions)
    .values({ userId, ...s })
    .onConflictDoUpdate({ target: pushSubscriptions.endpoint, set: { userId, p256dh: s.p256dh, auth: s.auth, createdAt: sql`now()` } });
}

export const listSubscriptions = (db: Db, userId: string) => db.select().from(pushSubscriptions).where(eq(pushSubscriptions.userId, userId));

/** Turns push off for one device of mine. */
export async function removeSubscription(db: Db, userId: string, endpoint: string): Promise<void> {
  await db.delete(pushSubscriptions).where(and(eq(pushSubscriptions.userId, userId), eq(pushSubscriptions.endpoint, endpoint)));
}

/** Drops devices the push service says are gone (404/410). Not user-scoped: the endpoint came from our own row. */
export async function removeEndpoints(db: Db, endpoints: string[]): Promise<void> {
  if (endpoints.length) await db.delete(pushSubscriptions).where(inArray(pushSubscriptions.endpoint, endpoints));
}

/** Users with at least one push device, with the time zone and last digest day the cron needs. */
export const listDigestCandidates = (db: Db) =>
  db
    .selectDistinct({ id: users.id, tz: users.tz, digestOn: users.digestOn })
    .from(users)
    .innerJoin(pushSubscriptions, eq(pushSubscriptions.userId, users.id));

/** Claims today's digest for a user. True only for the one caller that gets to send it. */
export async function claimDigest(db: Db, userId: string, day: string): Promise<boolean> {
  const rows = await db
    .update(users)
    .set({ digestOn: day })
    .where(and(eq(users.id, userId), sql`(${users.digestOn} is null or ${users.digestOn} < ${day})`))
    .returning({ id: users.id });
  return rows.length > 0;
}
