"use server";

import { auth } from "@/lib/auth";
import { getUserIdOrNull, requireUserId } from "@/lib/auth/session";
import { getDb } from "@/lib/db/client";
import * as q from "@/lib/db/queries";
import { parsePaymentDetails } from "@/lib/validation/payment";
import type { PaymentDetails } from "@/types/domain";

/** The signed-in user's public profile, or null for guests. */
export async function getCurrentUser() {
  const u = (await auth())?.user;
  if (!u?.id) return null;
  return { id: u.id, name: u.name ?? null, email: u.email ?? null, image: u.image ?? null };
}

/** The signed-in user's payment details, or null (guests keep theirs on the device, see lib/payment/store.ts). */
export async function getPaymentDetails(): Promise<PaymentDetails | null> {
  const userId = await getUserIdOrNull();
  if (!userId) return null;
  const stored = await q.getPaymentDetails(getDb(), userId);
  if (!stored) return null;
  // Re-check what comes back from the database too, so a bad row can never reach friends' messages.
  const parsed = parsePaymentDetails(stored);
  return parsed.ok && Object.keys(parsed.value).length ? parsed.value : null;
}

export type SavePaymentResult = { ok: true; details: PaymentDetails | null } | { ok: false; error: string };

/** Validates, then saves (an empty object clears). The user id comes from the session, never the client. */
export async function savePaymentDetails(input: unknown): Promise<SavePaymentResult> {
  const userId = await requireUserId();
  const parsed = parsePaymentDetails(input);
  if (!parsed.ok) return { ok: false, error: parsed.error };
  return { ok: true, details: await q.setPaymentDetails(getDb(), userId, parsed.value) };
}
