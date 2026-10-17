import { timingSafeEqual } from "node:crypto";

import { getDb } from "@/lib/db/client";
import { balancesFor } from "@/lib/db/balances";
import { unreadCount } from "@/lib/db/notifications";
import { claimDigest, listDigestCandidates } from "@/lib/db/push";
import { listPendingClaimsForOwner } from "@/lib/db/social";
import { buildDigest, DIGEST_HOUR, localDayHour } from "@/lib/push/digest";
import { pushConfigured, pushToUser } from "@/lib/push/send";

// Vercel Cron calls this every hour with `Authorization: Bearer $CRON_SECRET`. Each opted-in user gets one
// summary a day, from DIGEST_HOUR in their own time zone; users.digest_on makes a repeat call a no-op.

export const dynamic = "force-dynamic";

function authorized(req: Request): boolean {
  const secret = process.env.CRON_SECRET;
  if (!secret) return false;
  const got = Buffer.from(req.headers.get("authorization") ?? "");
  const want = Buffer.from(`Bearer ${secret}`);
  return got.length === want.length && timingSafeEqual(got, want);
}

export async function GET(req: Request) {
  if (!authorized(req)) return new Response("Unauthorized", { status: 401 });
  if (!pushConfigured()) return Response.json({ sent: 0, skipped: "push not configured" });

  const db = getDb();
  const now = new Date();
  const due = (await listDigestCandidates(db)).flatMap((u) => {
    const { day, hour } = localDayHour(u.tz, now);
    return hour >= DIGEST_HOUR && (!u.digestOn || u.digestOn < day) ? [{ id: u.id, day }] : [];
  });

  let sent = 0;
  for (let i = 0; i < due.length; i += 10) {
    const results = await Promise.all(
      due.slice(i, i + 10).map(async ({ id, day }): Promise<number> => {
        try {
          const [bal, unread, claims] = await Promise.all([balancesFor(db, id, day), unreadCount(db, id), listPendingClaimsForOwner(db, id)]);
          const digest = buildDigest({ people: bal.people, unread, claimsToConfirm: claims.length });
          if (!digest || !(await claimDigest(db, id, day))) return 0;
          return (await pushToUser(db, id, { ...digest, tag: `digest:${day}` })) > 0 ? 1 : 0;
        } catch {
          return 0;
        }
      }),
    );
    sent += results.reduce((a, b) => a + b, 0);
  }
  return Response.json({ due: due.length, sent });
}
