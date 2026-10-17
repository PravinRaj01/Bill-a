import { inArray, sql } from "drizzle-orm";

import { dateInZone, digestSummary } from "@/lib/notifications/digest";
import { computeBalances } from "./balances";
import type { Db } from "./client";
import { notify } from "./notifications";
import { billHistory, billShares, users } from "./schema";

export interface DigestRun {
  considered: number;
  sent: number;
}

/**
 * Sends today's digest to everyone who has something to settle. "Today" is each person's own day (their time
 * zone), and the dedupe key is that date, so running this twice (a retry, a double cron) sends one. A failure for
 * one person never stops the others.
 */
export async function runDigest(db: Db, now: Date = new Date()): Promise<DigestRun> {
  const [tracked, shared] = await Promise.all([
    db.selectDistinct({ id: billHistory.userId }).from(billHistory).where(sql`${billHistory.data} -> 'settlement' is not null`),
    db.selectDistinct({ id: billShares.userId }).from(billShares),
  ]);
  const ids = [...new Set([...tracked, ...shared].map((r) => r.id))];
  if (ids.length === 0) return { considered: 0, sent: 0 };
  const people = await db.select({ id: users.id, tz: users.tz }).from(users).where(inArray(users.id, ids));

  let sent = 0;
  for (const u of people) {
    try {
      const today = dateInZone(now, u.tz);
      const summary = digestSummary((await computeBalances(db, u.id, today)).people, today);
      if (!summary) continue;
      if (await notify(db, { userId: u.id, kind: "digest", payload: { summary }, dedupe: `digest:${today}` })) sent++;
    } catch {
      /* skip this person, carry on */
    }
  }
  return { considered: people.length, sent };
}
