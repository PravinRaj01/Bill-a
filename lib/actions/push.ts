"use server";

import { z } from "zod";
import { getUserIdOrNull, requireUserId } from "@/lib/auth/session";
import { getDb } from "@/lib/db/client";
import * as p from "@/lib/db/push";
import * as q from "@/lib/db/queries";
import { CATEGORIES, type NotifyPrefs } from "@/lib/notifications/prefs";
import { pushConfigured, pushToUser } from "@/lib/push/send";

// Turning phone notifications on and off. The user comes from the session only; the browser sends the
// subscription it just made, and only ever to attach it to the signed-in user.

const subscriptionSchema = z.object({
  endpoint: z.url().max(2048).refine((u) => u.startsWith("https://"), "https only"),
  keys: z.object({ p256dh: z.string().min(1).max(256), auth: z.string().min(1).max(64) }),
});

const categorySchema = z.enum(CATEGORIES);

/** Which kinds of notification reach my phone. Guests have none to set. */
export async function getNotifyPrefs(): Promise<NotifyPrefs> {
  const userId = await getUserIdOrNull();
  return userId ? q.getNotifyPrefs(getDb(), userId) : {};
}

export async function setNotifyPref(category: string, on: boolean): Promise<NotifyPrefs> {
  const userId = await requireUserId();
  return q.setNotifyPref(getDb(), userId, categorySchema.parse(category), z.boolean().parse(on));
}

export interface PushState {
  /** The server has VAPID keys, so push can work at all. */
  available: boolean;
  publicKey: string | null;
}

export async function getPushState(): Promise<PushState> {
  const userId = await getUserIdOrNull();
  const available = !!userId && pushConfigured();
  return { available, publicKey: available ? (process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY ?? null) : null };
}

export async function savePushSubscription(input: unknown): Promise<void> {
  const userId = await requireUserId();
  const s = subscriptionSchema.parse(input);
  await p.saveSubscription(getDb(), userId, { endpoint: s.endpoint, p256dh: s.keys.p256dh, auth: s.keys.auth });
}

export async function removePushSubscription(endpoint: string): Promise<boolean> {
  const userId = await requireUserId();
  return p.removeSubscription(getDb(), userId, z.string().min(1).max(2048).parse(endpoint));
}

/** "Send me a test": a real push to my own devices, so I can see it works. Returns how many devices got it and how long the server took to hand it over. */
export async function sendTestPush(): Promise<{ delivered: number; ms: number }> {
  const userId = await requireUserId();
  const started = Date.now();
  const delivered = await pushToUser(getDb(), userId, { urgent: true, force: true, title: "Bill.a", body: "Notifications are on. You'll hear about friends, shared bills and payments here.", url: "/dashboard/notifications", tag: "test" });
  return { delivered, ms: Date.now() - started };
}
