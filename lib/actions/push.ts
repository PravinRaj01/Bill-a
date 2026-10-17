"use server";

import { z } from "zod";
import { requireUserId } from "@/lib/auth/session";
import { getDb } from "@/lib/db/client";
import * as p from "@/lib/db/push";
import { pushToUser } from "@/lib/push/send";

// Turning phone notifications on and off. The user comes from the session only; the browser sends the
// subscription it just made, and only ever to attach it to the signed-in user.

const subscriptionSchema = z.object({
  endpoint: z.url().max(2048).refine((u) => u.startsWith("https://"), "https only"),
  keys: z.object({ p256dh: z.string().min(1).max(256), auth: z.string().min(1).max(64) }),
});

export async function savePushSubscription(input: unknown): Promise<void> {
  const userId = await requireUserId();
  const s = subscriptionSchema.parse(input);
  await p.saveSubscription(getDb(), userId, { endpoint: s.endpoint, p256dh: s.keys.p256dh, auth: s.keys.auth });
}

export async function removePushSubscription(endpoint: string): Promise<boolean> {
  const userId = await requireUserId();
  return p.removeSubscription(getDb(), userId, z.string().min(1).max(2048).parse(endpoint));
}

/** "Send me a test": a real push to my own devices, so I can see it works. Returns how many devices got it. */
export async function sendTestPush(): Promise<number> {
  const userId = await requireUserId();
  return pushToUser(getDb(), userId, { title: "Bill.a", body: "Notifications are on. You'll hear about friends, shared bills and payments here.", url: "/dashboard/notifications", tag: "test" });
}
