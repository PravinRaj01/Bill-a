"use server";

import { auth } from "@/lib/auth";

/** The signed-in user's public profile, or null for guests. */
export async function getCurrentUser() {
  const u = (await auth())?.user;
  if (!u?.id) return null;
  return { id: u.id, name: u.name ?? null, email: u.email ?? null, image: u.image ?? null };
}
