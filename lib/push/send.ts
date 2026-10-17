import webpush from "web-push";

import type { Db } from "@/lib/db/client";
import { listSubscriptions, removeEndpoints } from "@/lib/db/push";

// Web Push to a user's opted-in devices. Everything here is best effort: a push that fails must never fail the
// action that caused it (saving a bill, accepting a friend), so nothing in this file throws.

export interface PushPayload {
  title: string;
  body: string;
  /** Where tapping it goes (a path on this site). */
  url: string;
  /** Same tag replaces the earlier banner instead of stacking. */
  tag?: string;
}

export function pushConfigured(): boolean {
  return !!(process.env.VAPID_PUBLIC_KEY && process.env.VAPID_PRIVATE_KEY);
}

export const vapidPublicKey = () => process.env.VAPID_PUBLIC_KEY ?? null;

let configured = false;
function configure() {
  if (configured) return;
  webpush.setVapidDetails(process.env.VAPID_SUBJECT || "mailto:hello@example.invalid", process.env.VAPID_PUBLIC_KEY!, process.env.VAPID_PRIVATE_KEY!);
  configured = true;
}

/** Sends to every device of one user and forgets the ones the push service says are gone. Returns how many were delivered. */
export async function pushToUser(db: Db, userId: string, payload: PushPayload): Promise<number> {
  if (!pushConfigured()) return 0;
  try {
    configure();
    const subs = await listSubscriptions(db, userId);
    if (subs.length === 0) return 0;
    const body = JSON.stringify(payload);
    const dead: string[] = [];
    let sent = 0;
    await Promise.all(
      subs.map(async (s) => {
        try {
          await webpush.sendNotification({ endpoint: s.endpoint, keys: { p256dh: s.p256dh, auth: s.auth } }, body, { TTL: 60 * 60 * 24, timeout: 8000 });
          sent++;
        } catch (e) {
          const status = (e as { statusCode?: number }).statusCode;
          if (status === 404 || status === 410) dead.push(s.endpoint);
        }
      }),
    );
    await removeEndpoints(db, dead);
    return sent;
  } catch {
    return 0;
  }
}
