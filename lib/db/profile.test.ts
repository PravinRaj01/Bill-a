import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { config } from "dotenv";
import { eq, inArray, sql } from "drizzle-orm";
import { getDb } from "./client";
import { users } from "./schema";
import { getProfile, setProfile, setTimeZone, setUserName } from "./queries";

// The profile (display name, @username, portrait, time zone) is keyed by the session's user id only. Real Neon
// DB (the test branch, never .env.local); skipped without .env.test.local. Throwaway users are removed afterwards.
config({ path: ".env.test.local", quiet: true });
const hasDb = !!process.env.DATABASE_URL;

describe.skipIf(!hasDb)("profile (real Neon DB)", { timeout: 60_000 }, () => {
  const db = getDb();
  const tag = crypto.randomUUID().slice(0, 8);
  let A = "";
  let B = "";
  const nameOf = async (id: string) => (await db.select({ name: users.name }).from(users).where(eq(users.id, id)))[0]?.name ?? null;

  beforeAll(async () => {
    const [a, b] = await db
      .insert(users)
      .values([
        { email: `prof-a-${tag}@example.invalid`, name: "A" },
        { email: `prof-b-${tag}@example.invalid`, name: "B" },
      ])
      .returning({ id: users.id });
    A = a.id;
    B = b.id;
  });

  afterAll(async () => {
    await db.delete(users).where(inArray(users.id, [A, B].filter(Boolean)));
  });

  it("changes only the caller's name", async () => {
    expect(await setUserName(db, A, "Alexis Reed")).toBe("Alexis Reed");
    expect(await nameOf(A)).toBe("Alexis Reed");
    expect(await nameOf(B)).toBe("B");
  });

  it("null clears the name; an unknown id changes nothing", async () => {
    await setUserName(db, A, null);
    expect(await nameOf(A)).toBeNull();
    await setUserName(db, crypto.randomUUID(), "Ghost");
    expect(await nameOf(A)).toBeNull();
    expect(await nameOf(B)).toBe("B");
  });

  it("starts with no username or portrait and the default time zone", async () => {
    expect(await getProfile(db, A)).toEqual({ username: null, avatarSeed: null, tz: "Asia/Kuala_Lumpur" });
  });

  it("sets a username and portrait for the caller only", async () => {
    const r = await setProfile(db, A, { username: `alex_${tag}`, avatarSeed: "sun-1" });
    expect(r).toEqual({ ok: true, profile: { username: `alex_${tag}`, avatarSeed: "sun-1", tz: "Asia/Kuala_Lumpur" } });
    expect((await getProfile(db, B))?.username).toBeNull();
  });

  it("refuses a username someone else has: taken, and nothing else changed", async () => {
    const r = await setProfile(db, B, { username: `alex_${tag}`, avatarSeed: "moon-2" });
    expect(r).toEqual({ ok: false, reason: "username-taken" });
    expect((await getProfile(db, B))?.avatarSeed).toBeNull();
  });

  it("two people claiming the same name at once: exactly one wins", async () => {
    const name = `race_${tag}`;
    const results = await Promise.all([setProfile(db, A, { username: name }), setProfile(db, B, { username: name })]);
    expect(results.filter((r) => r.ok)).toHaveLength(1);
    expect(results.filter((r) => !r.ok)).toEqual([{ ok: false, reason: "username-taken" }]);
  });

  it("changing your own username to the same value is fine, and null releases it", async () => {
    const mine = (await getProfile(db, A))?.username;
    expect(mine).not.toBeNull();
    expect((await setProfile(db, A, { username: mine })).ok).toBe(true);
    expect((await setProfile(db, A, { username: null })).ok).toBe(true);
    expect((await setProfile(db, B, { username: mine })).ok).toBe(true); // now free
  });

  it("an update with nothing in it leaves the profile alone; an unknown user is not found", async () => {
    const before = await getProfile(db, B);
    expect(await setProfile(db, B, {})).toEqual({ ok: true, profile: before });
    expect(await setProfile(db, crypto.randomUUID(), { avatarSeed: "x" })).toEqual({ ok: false, reason: "not-found" });
  });

  it("the database itself refuses a malformed username, even if the app code were bypassed", async () => {
    await expect(db.update(users).set({ username: "Has Space" }).where(eq(users.id, A))).rejects.toThrow();
    await expect(db.update(users).set({ username: "ab" }).where(eq(users.id, A))).rejects.toThrow();
    await expect(db.execute(sql`update "user" set username = 'UPPER' where id = ${A}`)).rejects.toThrow();
  });

  it("stores a time zone for the caller", async () => {
    await setTimeZone(db, A, "Asia/Singapore");
    expect((await getProfile(db, A))?.tz).toBe("Asia/Singapore");
    expect((await getProfile(db, B))?.tz).toBe("Asia/Kuala_Lumpur");
  });
});
