import { describe, expect, it } from "vitest";
import { isUrgent, notificationHref, notificationText, notificationTitle } from "./text";

const bill = "11111111-1111-4111-8111-111111111111";

describe("notification text", () => {
  it("builds every sentence from the kind and payload", () => {
    const p = { title: "Dinner out", amount: 3000, currency: "RM" };
    expect(notificationText({ kind: "friend_request", billId: null, payload: {} }, "Riley")).toBe("Riley sent you a friend request");
    expect(notificationText({ kind: "friend_accepted", billId: null, payload: {} }, "@riley")).toBe("@riley accepted your friend request");
    expect(notificationText({ kind: "bill_added", billId: bill, payload: p }, "Alex")).toBe('Alex added you to "Dinner out"');
    expect(notificationText({ kind: "claim_made", billId: bill, payload: p }, "Riley")).toBe('Riley says they paid you RM30.00 for "Dinner out"');
    expect(notificationText({ kind: "claim_confirmed", billId: bill, payload: p }, "Alex")).toBe('Alex confirmed your RM30.00 payment for "Dinner out"');
    expect(notificationText({ kind: "claim_rejected", billId: bill, payload: p }, "Alex")).toBe('Alex says your RM30.00 for "Dinner out" hasn\'t arrived');
    expect(notificationText({ kind: "bill_deleted", billId: null, payload: { title: "Dinner out" } }, "Alex")).toBe('Alex deleted "Dinner out"');
    expect(notificationText({ kind: "welcome", billId: null, payload: {} }, "Bill.a")).toContain("AI key");
    expect(notificationHref({ kind: "welcome", billId: null, payload: {} })).toBe("/dashboard/account?tab=ai&guide=1");
    expect(notificationText({ kind: "bill_unshared", billId: null, payload: { title: "Dinner out" } }, "Alex")).toBe('Alex stopped sharing "Dinner out" with you');
    expect(notificationText({ kind: "claim_changed", billId: bill, payload: p }, "Alex")).toContain('changed "Dinner out"');
    expect(notificationText({ kind: "nudge", billId: bill, payload: p }, "Alex")).toBe('Alex is reminding you about RM30.00 on "Dinner out"');
  });

  it("copes with a missing title or amount", () => {
    expect(notificationText({ kind: "bill_deleted", billId: null, payload: {} }, "Alex")).toBe("Alex deleted a bill");
  });

  it("sends each kind where it belongs, and nowhere for a deleted bill", () => {
    expect(notificationHref({ kind: "friend_request", billId: null, payload: {} })).toBe("/dashboard/people");
    expect(notificationHref({ kind: "bill_added", billId: bill, payload: {} })).toBe(`/dashboard/shared/${bill}`);
    expect(notificationHref({ kind: "claim_made", billId: bill, payload: {} })).toBe(`/dashboard/history/${bill}`);
    expect(notificationHref({ kind: "nudge", billId: null, payload: {} })).toBeNull();
    expect(notificationHref({ kind: "bill_deleted", billId: null, payload: {} })).toBeNull();
  });
});

describe("phone notification titles", () => {
  const kinds = ["friend_request", "friend_accepted", "bill_added", "bill_unshared", "bill_deleted", "claim_made", "claim_confirmed", "claim_rejected", "claim_changed", "nudge", "digest", "welcome"] as const;
  it("every kind has its own short headline", () => {
    const titles = kinds.map((k) => notificationTitle(k));
    expect(new Set(titles).size).toBe(kinds.length);
    expect(titles.every((t) => t.length > 0 && t.length <= 24)).toBe(true);
    expect(notificationTitle("claim_made")).toBe("Payment");
  });
  it("the daily summary and the welcome are not urgent; things someone is waiting on are", () => {
    expect(isUrgent("digest")).toBe(false);
    expect(isUrgent("welcome")).toBe(false);
    expect(isUrgent("friend_request")).toBe(true);
    expect(isUrgent("claim_made")).toBe(true);
  });
});
