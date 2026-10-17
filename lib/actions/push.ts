"use server";

import { z } from "zod";
import { getUserIdOrNull, requireUserId } from "@/lib/auth/session";
import { getDb } from "@/lib/db/client";
import * as p from "@/lib/db/push";
import { pushConfigured, vapidPublicKey } from "@/lib/push/send";

// Turning push on and off for THIS device. The user comes from the session only; an endpoint from the browser
// is only ever stored for, or removed from, that user.

export interface PushState {
  /** The server has VAPID keys, so push can work at all. */
  available: boolean;
  publicKey: string | null;
  /** How many of my devices are subscribed (this device's own state comes from the browser). */
  devices: number;
}

export async function getPushState(): Promise<PushState> {
  const userId = await getUserIdOrNull();
  const available = pushConfigured();
  if (!userId || !available) return { available, publicKey: null, devices: 0 };
  return { available, publicKey: vapidPublicKey(), devices: (await p.listSubscriptions(getDb(), userId)).length };
}

const subscription = z.object({
  endpoint: z.url({ protocol: /^https$/ }).max(2048),
  keys: z.object({ p256dh: z.string().min(1).max(256), auth: z.string().min(1).max(256) }),
});

export async function savePushSubscription(input: unknown): Promise<void> {
  const userId = await requireUserId();
  const s = subscription.parse(input);
  await p.saveSubscription(getDb(), userId, { endpoint: s.endpoint, p256dh: s.keys.p256dh, auth: s.keys.auth });
}

export async function removePushSubscription(endpoint: string): Promise<void> {
  const userId = await requireUserId();
  await p.removeSubscription(getDb(), userId, z.string().max(2048).parse(endpoint));
}
