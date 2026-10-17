import { z } from "zod";

// A person's public profile: an @username (how friends find them, later) and a portrait seed. Both are shown
// to other people, so both are tightly bounded.

/** Names we keep for ourselves or that would confuse (routes, roles, the app's own name). */
export const RESERVED_USERNAMES = new Set([
  "admin", "administrator", "root", "support", "help", "staff", "team", "official", "moderator", "mod", "owner", "system",
  "billa", "bill", "bill_a", "billapp", "app", "api", "www", "me", "you", "null", "undefined", "test",
  "login", "logout", "signup", "signin", "account", "settings", "dashboard", "history", "balances", "groups",
  "friends", "people", "add", "invite", "security", "privacy", "terms", "about",
]);

/** Lower-cased, 3 to 20 of a-z 0-9 _, and not reserved. */
export const usernameSchema = z
  .string()
  .trim()
  .toLowerCase()
  .regex(/^[a-z0-9_]{3,20}$/, "Use 3 to 20 letters, numbers or underscores")
  .refine((u) => !RESERVED_USERNAMES.has(u), "That username isn't available");

/** The text a portrait is generated from. Plain characters only; never trusted as markup. */
export const avatarSeedSchema = z
  .string()
  .trim()
  .min(1)
  .max(40)
  .regex(/^[A-Za-z0-9 _.-]+$/, "Invalid portrait");

// null clears, undefined leaves alone.
export const profileInputSchema = z
  .object({
    username: usernameSchema.nullable().optional(),
    avatarSeed: avatarSeedSchema.nullable().optional(),
  })
  .strict();

export type ProfileInput = z.infer<typeof profileInputSchema>;

/** The first problem with a profile input as a short message, or the cleaned input. */
export function parseProfileInput(input: unknown): { ok: true; value: ProfileInput } | { ok: false; error: string } {
  const r = profileInputSchema.safeParse(input);
  return r.success ? { ok: true, value: r.data } : { ok: false, error: r.error.issues[0]?.message ?? "Invalid profile" };
}

/** An IANA time zone name the runtime knows ("Asia/Kuala_Lumpur"), or null. */
export function validTimeZone(tz: unknown): string | null {
  if (typeof tz !== "string" || tz.length > 64 || !/^[A-Za-z0-9_+\-/]+$/.test(tz)) return null;
  try {
    new Intl.DateTimeFormat("en", { timeZone: tz });
    return tz;
  } catch {
    return null;
  }
}
