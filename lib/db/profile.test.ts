import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { config } from "dotenv";
import { eq, inArray } from "drizzle-orm";
import { getDb } from "./client";
import { users } from "./schema";
import { setUserName } from "./queries";

// The display name is keyed by the session's user id only. Real Neon DB (the test branch, never
// .env.local); skipped without .env.test.local. Throwaway users are removed afterwards.
config({ path: ".env.test.local", quiet: true });
const hasDb = !!process.env.DATABASE_URL;

describe.skipIf(!hasDb)("profile name (real Neon DB)", { timeout: 60_000 }, () => {
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

  it("null clears it", async () => {
    await setUserName(db, A, null);
    expect(await nameOf(A)).toBeNull();
    expect(await nameOf(B)).toBe("B");
  });

  it("an unknown id changes nothing", async () => {
    await setUserName(db, crypto.randomUUID(), "Ghost");
    expect(await nameOf(A)).toBeNull();
    expect(await nameOf(B)).toBe("B");
  });
});
