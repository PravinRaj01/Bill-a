import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { config } from "dotenv";
import { eq, inArray } from "drizzle-orm";
import { getDb } from "./client";
import { savedGroups, users } from "./schema";
import { createGroup, listGroups } from "./queries";

// "Best Couple" x2: saving a group name the user already has must UPDATE that group,
// never create a lookalike. Real Neon DB (the guarantee is a unique constraint, so
// mocking the DB would prove nothing). Skipped when DATABASE_URL isn't set.

config({ path: ".env.local" });
const hasDb = !!process.env.DATABASE_URL;

describe.skipIf(!hasDb)("saving a group never duplicates it (real Neon DB)", { timeout: 60_000 }, () => {
  const db = getDb();
  const tag = crypto.randomUUID().slice(0, 8);
  let A = "";
  let B = "";

  beforeAll(async () => {
    const [a, b] = await db
      .insert(users)
      .values([
        { email: `grp-a-${tag}@example.invalid`, name: "A" },
        { email: `grp-b-${tag}@example.invalid`, name: "B" },
      ])
      .returning({ id: users.id });
    A = a.id;
    B = b.id;
  });

  afterAll(async () => {
    // cascade removes their groups
    await db.delete(users).where(inArray(users.id, [A, B].filter(Boolean)));
  });

  it("saving the same name twice yields ONE group, with the latest members", async () => {
    const first = await createGroup(db, A, { groupName: "Best Couple", names: ["Pravin", "Wifey"] });
    const second = await createGroup(db, A, { groupName: "Best Couple", names: ["Pravin", "Wifey", "Sam"] });
    expect(second.id).toBe(first.id);
    expect(second.names).toEqual(["Pravin", "Wifey", "Sam"]);
    expect((await listGroups(db, A)).filter((g) => g.groupName === "Best Couple")).toHaveLength(1);
  });

  it("ignores case and surrounding spaces, and keeps the original display name", async () => {
    const original = (await listGroups(db, A)).find((g) => g.groupName === "Best Couple")!;
    const again = await createGroup(db, A, { groupName: "  best couple ", names: ["X", "Y"] });
    expect(again.id).toBe(original.id);
    expect(again.groupName).toBe("Best Couple");
    expect(again.names).toEqual(["X", "Y"]);
    expect(await db.select().from(savedGroups).where(eq(savedGroups.userId, A))).toHaveLength(1);
  });

  it("two saves at the same instant (a double tap) still produce one group", async () => {
    const results = await Promise.all([
      createGroup(db, A, { groupName: "Race Crew", names: ["A", "B"] }),
      createGroup(db, A, { groupName: "Race Crew", names: ["A", "B"] }),
      createGroup(db, A, { groupName: "race crew", names: ["A", "B"] }),
    ]);
    expect(new Set(results.map((r) => r.id)).size).toBe(1);
    expect((await listGroups(db, A)).filter((g) => g.groupName.toLowerCase() === "race crew")).toHaveLength(1);
  });

  it("different names are different groups", async () => {
    await createGroup(db, A, { groupName: "Work Lunch", names: ["A"] });
    const names = (await listGroups(db, A)).map((g) => g.groupName).sort();
    expect(names).toEqual(["Best Couple", "Race Crew", "Work Lunch"]);
  });

  it("another user can use the same name: uniqueness is per user, and never touches their group", async () => {
    const mine = (await listGroups(db, A)).find((g) => g.groupName === "Best Couple")!;
    const theirs = await createGroup(db, B, { groupName: "Best Couple", names: ["B-only"] });
    expect(theirs.id).not.toBe(mine.id);
    expect((await listGroups(db, A)).find((g) => g.groupName === "Best Couple")!.names).toEqual(["X", "Y"]);
    expect((await listGroups(db, B)).map((g) => g.names)).toEqual([["B-only"]]);
  });

  it("the database itself refuses a lookalike, even if the app code were bypassed", async () => {
    await expect(
      db.insert(savedGroups).values({ userId: A, groupName: "BEST COUPLE", names: ["Z"] }),
    ).rejects.toThrow();
  });
});
