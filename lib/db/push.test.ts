import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { config } from "dotenv";
import { inArray } from "drizzle-orm";
import { getDb } from "./client";
import { users } from "./schema";
import { claimDigest, listDigestCandidates, listSubscriptions, removeEndpoints, removeSubscription, saveSubscription } from "./push";

// Push devices and the once-a-day digest claim, on the real Neon test branch.
config({ path: ".env.test.local", quiet: true });
const hasDb = !!process.env.DATABASE_URL;

describe.skipIf(!hasDb)("push subscriptions (real Neon DB)", { timeout: 60_000 }, () => {
  const db = getDb();
  const tag = crypto.randomUUID().slice(0, 8);
  let A = "";
  let B = "";
  const ep = (n: string) => `https://push.example.invalid/${tag}/${n}`;

  beforeAll(async () => {
    const rows = await db
      .insert(users)
      .values([{ email: `push-a-${tag}@example.invalid`, name: "A" }, { email: `push-b-${tag}@example.invalid`, name: "B" }])
      .returning({ id: users.id, name: users.name });
    A = rows.find((r) => r.name === "A")!.id;
    B = rows.find((r) => r.name === "B")!.id;
  });
  afterAll(async () => {
    await db.delete(users).where(inArray(users.id, [A, B].filter(Boolean)));
  });

  it("saves per device, replaces on re-subscribe, and moves a device to the new account", async () => {
    await saveSubscription(db, A, { endpoint: ep("1"), p256dh: "k1", auth: "a1" });
    await saveSubscription(db, A, { endpoint: ep("1"), p256dh: "k2", auth: "a2" });
    expect((await listSubscriptions(db, A)).map((s) => s.p256dh)).toEqual(["k2"]);
    await saveSubscription(db, B, { endpoint: ep("1"), p256dh: "k3", auth: "a3" });
    expect(await listSubscriptions(db, A)).toHaveLength(0);
    expect(await listSubscriptions(db, B)).toHaveLength(1);
  });

  it("only lets a user remove their own device; dead endpoints are dropped by endpoint", async () => {
    await removeSubscription(db, A, ep("1"));
    expect(await listSubscriptions(db, B)).toHaveLength(1);
    await saveSubscription(db, B, { endpoint: ep("2"), p256dh: "k", auth: "a" });
    await removeEndpoints(db, [ep("1"), ep("2")]);
    expect(await listSubscriptions(db, B)).toHaveLength(0);
  });

  it("lists only users with a device as digest candidates, and claims a day once", async () => {
    await saveSubscription(db, A, { endpoint: ep("3"), p256dh: "k", auth: "a" });
    const ids = (await listDigestCandidates(db)).map((u) => u.id);
    expect(ids).toContain(A);
    expect(ids).not.toContain(B);
    expect(await claimDigest(db, A, "2026-10-17")).toBe(true);
    expect(await claimDigest(db, A, "2026-10-17")).toBe(false);
    expect(await claimDigest(db, A, "2026-10-16")).toBe(false);
    expect(await claimDigest(db, A, "2026-10-18")).toBe(true);
  });
});
