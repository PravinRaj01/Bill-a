import type { NotificationKind } from "@/lib/db/schema";

// What reaches the phone. Every kind of notification belongs to one category, and a person can switch a category
// off for push. The inbox always keeps its record; the one exception is the daily summary, where "off" means it is
// not made at all. Only categories switched OFF are stored, so a new category is on for everyone until they choose.

export const CATEGORIES = ["friends", "bills", "payments", "reminders", "summary"] as const;
export type Category = (typeof CATEGORIES)[number];
export type NotifyPrefs = Partial<Record<Category, boolean>>;

export const CATEGORY_INFO: Record<Category, { title: string; blurb: string }> = {
  friends: { title: "Friend requests", blurb: "Someone asks to be pals, or accepts your request" },
  bills: { title: "Shared bills", blurb: "A friend adds you to a bill, takes you off it, or deletes it" },
  payments: { title: "Payments", blurb: "A friend says they paid, or you hear your payment arrived" },
  reminders: { title: "Reminders", blurb: "A friend nudges you about what you owe" },
  summary: { title: "Daily summary", blurb: "One short note a day when money is still owed" },
};

/** The category a notification belongs to, or null when it always goes through (the welcome note). */
export function kindCategory(kind: NotificationKind): Category | null {
  switch (kind) {
    case "friend_request":
    case "friend_accepted":
      return "friends";
    case "bill_added":
    case "bill_unshared":
    case "bill_deleted":
      return "bills";
    case "claim_made":
    case "claim_confirmed":
    case "claim_rejected":
    case "claim_changed":
      return "payments";
    case "nudge":
      return "reminders";
    case "digest":
      return "summary";
    case "welcome":
      return null;
  }
}

/** Whatever the database holds, reduced to known categories set to a boolean (anything else is dropped). */
export function parsePrefs(raw: unknown): NotifyPrefs {
  const out: NotifyPrefs = {};
  if (raw && typeof raw === "object" && !Array.isArray(raw)) {
    for (const c of CATEGORIES) {
      const v = (raw as Record<string, unknown>)[c];
      if (typeof v === "boolean") out[c] = v;
    }
  }
  return out;
}

export const isCategoryOn = (prefs: NotifyPrefs, category: Category): boolean => prefs[category] !== false;

/** Should this kind reach the phone for someone with these preferences? */
export function wantsPush(prefs: NotifyPrefs, kind: NotificationKind): boolean {
  const c = kindCategory(kind);
  return c === null || isCategoryOn(prefs, c);
}
