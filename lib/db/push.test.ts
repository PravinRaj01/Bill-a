import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { config } from "dotenv";
import { inArray } from "drizzle-orm";
import { getDb } from "./client";
import { users } from "./schema";
import { dropDeadSubscription, listSubscriptions, removeSubscription, saveSubscription } from "./push";
import { pushToUser } from "@/lib/push/send";

// Push devices on the real Neon test branch, including the cross-user cases.
config({ path: ".env.test.local", quiet: true });
const hasDb = !!process.env.DATABASE_URL;

describe.skipIf(!hasDb)("push subscriptions (real Neon DB)", { timeout: 60_000 }, () => {
  const db = hasDb ? getDb() : (null as never);
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

  it("only lets a user remove their own device", async () => {
    expect(await removeSubscription(db, A, ep("1"))).toBe(false);
    expect(await listSubscriptions(db, B)).toHaveLength(1);
    await dropDeadSubscription(db, ep("1"));
    expect(await listSubscriptions(db, B)).toHaveLength(0);
  });

  it("pushes to every device and drops the ones the push service says are gone", async () => {
    await saveSubscription(db, A, { endpoint: ep("live"), p256dh: "k", auth: "a" });
    await saveSubscription(db, A, { endpoint: ep("gone"), p256dh: "k", auth: "a" });
    const sent: string[] = [];
    const urgency: (boolean | undefined)[] = [];
    const n = await pushToUser(db, A, { id: "n1", urgent: true, title: "t", body: "b", url: "https://evil.example", tag: "x" }, async (sub, payload, opts) => {
      if (sub.endpoint.endsWith("gone")) throw Object.assign(new Error("gone"), { statusCode: 410 });
      sent.push(payload);
      urgency.push(opts?.urgent);
    });
    expect(n).toBe(1);
    expect(JSON.parse(sent[0]).url).toBe("/dashboard/notifications");
    expect(JSON.parse(sent[0])).toMatchObject({ id: "n1", title: "t", tag: "x" });
    expect(JSON.parse(sent[0]).urgent).toBeUndefined(); // urgency is a delivery option, not part of what the phone shows
    expect(urgency).toEqual([true]);
    expect((await listSubscriptions(db, A)).map((s) => s.endpoint)).toEqual([ep("live")]);
  });
});
