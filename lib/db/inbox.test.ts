import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { config } from "dotenv";
import { inArray } from "drizzle-orm";
import { getDb } from "./client";
import { users } from "./schema";
import { dismiss, dismissAll, listNotifications, notify, restore, unreadCount } from "./notifications";
import { getNotifyPrefs, setNotifyPref } from "./queries";

// Clearing the inbox, on the real Neon test branch: only your own rows, still deduped, and undoable.
config({ path: ".env.test.local", quiet: true });
const hasDb = !!process.env.DATABASE_URL;

describe.skipIf(!hasDb)("clearing the inbox (real Neon DB)", { timeout: 60_000 }, () => {
  const db = hasDb ? getDb() : (null as never);
  const tag = crypto.randomUUID().slice(0, 8);
  let A = "";
  let B = "";

  beforeAll(async () => {
    const rows = await db
      .insert(users)
      .values([{ email: `inbox-a-${tag}@example.invalid`, name: "A" }, { email: `inbox-b-${tag}@example.invalid`, name: "B" }])
      .returning({ id: users.id, name: users.name });
    A = rows.find((r) => r.name === "A")!.id;
    B = rows.find((r) => r.name === "B")!.id;
  });
  afterAll(async () => {
    await db.delete(users).where(inArray(users.id, [A, B].filter(Boolean)));
  });

  it("clears one, reads it, hides it from the list and the count, and can bring it back", async () => {
    await notify(db, { userId: A, kind: "digest", payload: { summary: "x" }, dedupe: "d1" });
    await notify(db, { userId: A, kind: "digest", payload: { summary: "y" }, dedupe: "d2" });
    const [first] = await listNotifications(db, A);
    expect(await unreadCount(db, A)).toBe(2);

    expect(await dismiss(db, A, [first.id])).toEqual([first.id]);
    expect((await listNotifications(db, A)).map((n) => n.id)).not.toContain(first.id);
    expect(await unreadCount(db, A)).toBe(1); // clearing also reads it

    expect(await restore(db, A, [first.id])).toBe(1);
    expect((await listNotifications(db, A)).map((n) => n.id)).toContain(first.id);
  });

  it("only touches your own notifications", async () => {
    const [mine] = await listNotifications(db, A);
    expect(await dismiss(db, B, [mine.id])).toEqual([]); // not B's
    expect(await restore(db, B, [mine.id])).toBe(0);
    expect(await listNotifications(db, A)).toContainEqual(expect.objectContaining({ id: mine.id }));
  });

  it("a cleared notification does not come back when the same event is sent again", async () => {
    await notify(db, { userId: B, kind: "digest", payload: { summary: "z" }, dedupe: "once" });
    const cleared = await dismissAll(db, B);
    expect(cleared).toHaveLength(1);
    expect(await notify(db, { userId: B, kind: "digest", payload: { summary: "z" }, dedupe: "once" })).toBe(false); // a retry
    expect(await listNotifications(db, B)).toHaveLength(0);
    expect(await unreadCount(db, B)).toBe(0);
  });

  it("notification preferences: stored per user, only what is off, switch back on removes it", async () => {
    expect(await getNotifyPrefs(db, A)).toEqual({});
    expect(await setNotifyPref(db, A, "payments", false)).toEqual({ payments: false });
    expect(await setNotifyPref(db, A, "summary", false)).toEqual({ payments: false, summary: false });
    expect(await getNotifyPrefs(db, B)).toEqual({}); // someone else's are untouched
    expect(await setNotifyPref(db, A, "payments", true)).toEqual({ summary: false });
    await setNotifyPref(db, A, "summary", true);
    expect(await getNotifyPrefs(db, A)).toEqual({});
  });

  it("clear all returns what it cleared and only for the caller", async () => {
    const aBefore = (await listNotifications(db, A)).length;
    const done = await dismissAll(db, A);
    expect(done).toHaveLength(aBefore);
    expect(await listNotifications(db, A)).toHaveLength(0);
    expect(await restore(db, A, done)).toBe(aBefore);
  });
});
