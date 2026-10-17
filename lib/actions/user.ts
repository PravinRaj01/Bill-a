"use server";

import { z } from "zod";
import { auth, unstable_update } from "@/lib/auth";
import { getUserIdOrNull, requireUserId } from "@/lib/auth/session";
import { getDb } from "@/lib/db/client";
import * as q from "@/lib/db/queries";
import { parsePaymentDetails } from "@/lib/validation/payment";
import { parseProfileInput, validTimeZone } from "@/lib/validation/profile";
import type { PaymentDetails } from "@/types/domain";

/** The signed-in user's public profile, or null for guests. */
export async function getCurrentUser() {
  const u = (await auth())?.user;
  if (!u?.id) return null;
  return { id: u.id, name: u.name ?? null, email: u.email ?? null, image: u.image ?? null, username: u.username ?? null, avatarSeed: u.avatarSeed ?? null };
}

const nameSchema = z.string().trim().max(80);

export type UpdateProfileResult = { ok: true; username: string | null; avatarSeed: string | null } | { ok: false; error: string };

/**
 * Saves the signed-in user's @username and/or portrait, and refreshes the live session with them (the token is
 * made at sign-in). The user id comes from the session only; unknown keys are refused, so this can't write
 * anything else.
 */
export async function updateProfile(input: unknown): Promise<UpdateProfileResult> {
  const userId = await requireUserId();
  const parsed = parseProfileInput(input);
  if (!parsed.ok) return { ok: false, error: parsed.error };
  const r = await q.setProfile(getDb(), userId, parsed.value);
  if (!r.ok) return { ok: false, error: r.reason === "username-taken" ? "That username is taken." : "Couldn't find your account." };
  await unstable_update({ user: { username: r.profile.username, avatarSeed: r.profile.avatarSeed } });
  return { ok: true, username: r.profile.username, avatarSeed: r.profile.avatarSeed };
}

/** Remembers the browser's time zone (quietly, for the daily digest). Ignored if it isn't a real zone. */
export async function setTimezone(tz: string): Promise<void> {
  const userId = await getUserIdOrNull();
  const zone = validTimeZone(tz);
  if (!userId || !zone) return;
  await q.setTimeZone(getDb(), userId, zone);
}

/** Saves the signed-in user's display name (empty clears it) and refreshes the current session with it. */
export async function updateProfileName(input: string): Promise<{ ok: true; name: string | null } | { ok: false; error: string }> {
  const userId = await requireUserId();
  const parsed = nameSchema.safeParse(input);
  if (!parsed.success) return { ok: false, error: "That name is too long." };
  const name = await q.setUserName(getDb(), userId, parsed.data || null);
  await unstable_update({ user: { name } });
  return { ok: true, name };
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
