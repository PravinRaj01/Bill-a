import webpush from "web-push";

import type { Db } from "@/lib/db/client";
import { dropDeadSubscription, listSubscriptions } from "@/lib/db/push";

// Sending a Web Push to every device a user turned notifications on for. Needs the VAPID keys in the
// environment; without them nothing is sent (the inbox still works).

export interface PushMessage {
  /** The inbox row this is about: tapping the notification marks it read. */
  id?: string;
  /** A friend's request or a payment should wake the phone; a daily summary can wait. */
  urgent?: boolean;
  /** Raise a system notification even when the app is open in front (the "send me a test" button, which must prove that path). */
  force?: boolean;
  title: string;
  body: string;
  /** Where tapping it goes: a path on this site. */
  url: string;
  /** Same tag replaces the earlier notification instead of stacking. */
  tag: string;
}

export type Sender = (sub: { endpoint: string; keys: { p256dh: string; auth: string } }, payload: string, opts?: { urgent?: boolean }) => Promise<unknown>;

let configured: boolean | null = null;
function configure(): boolean {
  if (configured !== null) return configured;
  const pub = process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY;
  const priv = process.env.VAPID_PRIVATE_KEY;
  const subject = process.env.VAPID_SUBJECT;
  if (!pub || !priv || !subject) return (configured = false);
  try {
    webpush.setVapidDetails(subject, pub, priv);
    configured = true;
  } catch {
    configured = false;
  }
  return configured;
}

export const pushConfigured = (): boolean => configure();

const defaultSender: Sender = (sub, payload, opts) => webpush.sendNotification(sub, payload, { TTL: 60 * 60 * 24, urgency: opts?.urgent ? "high" : "normal" });

/** Only same-site paths: a notification can never send someone off to another site. */
const safeUrl = (u: string) => (u.startsWith("/") && !u.startsWith("//") ? u : "/dashboard/notifications");

/**
 * Pushes to all of a user's devices. A device the push service says is gone (404 or 410) is deleted; any other
 * failure is skipped (the inbox has it anyway). Returns how many devices accepted it.
 */
export async function pushToUser(db: Db, userId: string, msg: PushMessage, send: Sender = defaultSender): Promise<number> {
  if (send === defaultSender && !configure()) return 0;
  const subs = await listSubscriptions(db, userId);
  const { urgent, ...shown } = msg;
  const payload = JSON.stringify({ ...shown, url: safeUrl(msg.url) });
  let delivered = 0;
  await Promise.all(
    subs.map(async (s) => {
      try {
        await send({ endpoint: s.endpoint, keys: { p256dh: s.p256dh, auth: s.auth } }, payload, { urgent });
        delivered++;
      } catch (e) {
        const status = (e as { statusCode?: number }).statusCode;
        if (status === 404 || status === 410) await dropDeadSubscription(db, s.endpoint).catch(() => {});
      }
    }),
  );
  return delivered;
}
