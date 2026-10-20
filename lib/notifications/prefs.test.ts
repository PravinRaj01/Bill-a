import { describe, expect, it } from "vitest";

import { CATEGORIES, isCategoryOn, kindCategory, parsePrefs, wantsPush } from "./prefs";
import type { NotificationKind } from "@/lib/db/schema";

const KINDS: NotificationKind[] = ["friend_request", "friend_accepted", "bill_added", "bill_unshared", "bill_deleted", "claim_made", "claim_confirmed", "claim_rejected", "claim_changed", "nudge", "digest", "welcome"];

describe("notification preferences", () => {
  it("every kind belongs to a category, except the welcome note, and every category has a kind", () => {
    const cats = KINDS.map(kindCategory);
    expect(cats.filter((c) => c === null)).toHaveLength(1);
    expect(kindCategory("welcome")).toBeNull();
    expect(new Set(cats.filter(Boolean))).toEqual(new Set(CATEGORIES));
  });

  it("everything is on until switched off", () => {
    expect(KINDS.every((k) => wantsPush({}, k))).toBe(true);
  });

  it("switching a category off mutes its kinds only", () => {
    const prefs = { payments: false };
    expect(wantsPush(prefs, "claim_made")).toBe(false);
    expect(wantsPush(prefs, "claim_rejected")).toBe(false);
    expect(wantsPush(prefs, "friend_request")).toBe(true);
    expect(wantsPush(prefs, "digest")).toBe(true);
    expect(isCategoryOn(prefs, "payments")).toBe(false);
  });

  it("the welcome note always goes through", () => {
    const allOff = Object.fromEntries(CATEGORIES.map((c) => [c, false]));
    expect(wantsPush(allOff, "welcome")).toBe(true);
  });

  it("parsePrefs keeps only known categories set to a boolean", () => {
    expect(parsePrefs({ friends: false, summary: true, bogus: false, bills: "no", nested: { x: 1 } })).toEqual({ friends: false, summary: true });
    expect(parsePrefs(null)).toEqual({});
    expect(parsePrefs([1, 2])).toEqual({});
    expect(parsePrefs("x")).toEqual({});
  });
});
