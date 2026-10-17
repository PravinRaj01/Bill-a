import { describe, expect, it } from "vitest";
import { buildDigest, localDayHour } from "./digest";

describe("buildDigest", () => {
  it("says nothing when there is nothing to say", () => {
    expect(buildDigest({ people: [], unread: 0, claimsToConfirm: 0 })).toBeNull();
    expect(buildDigest({ people: [{ currency: "RM", net: 0 }], unread: 0, claimsToConfirm: 0 })).toBeNull();
  });

  it("totals what is owed each way per currency", () => {
    const d = buildDigest({
      people: [{ currency: "RM", net: 4500 }, { currency: "RM", net: 500 }, { currency: "RM", net: -1200 }, { currency: "SGD", net: 1000 }],
      unread: 0,
      claimsToConfirm: 0,
    });
    expect(d?.body).toBe("Friends owe you RM50.00 + SGD 10.00 · You owe RM12.00");
    expect(d?.url).toBe("/dashboard/balances");
  });

  it("points at the inbox when something needs attention", () => {
    const d = buildDigest({ people: [], unread: 3, claimsToConfirm: 1 });
    expect(d).toMatchObject({ body: "1 payment to confirm · 3 unread", url: "/dashboard/notifications" });
  });
});

describe("localDayHour", () => {
  const t = new Date("2026-10-17T18:30:00Z");
  it("uses the user's own day and hour", () => {
    expect(localDayHour("Asia/Kuala_Lumpur", t)).toEqual({ day: "2026-10-18", hour: 2 });
    expect(localDayHour("America/Los_Angeles", t)).toEqual({ day: "2026-10-17", hour: 11 });
  });
  it("falls back to UTC for a bad zone", () => {
    expect(localDayHour("Not/AZone", t)).toEqual({ day: "2026-10-17", hour: 18 });
  });
});
