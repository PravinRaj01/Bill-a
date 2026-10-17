import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { config } from "dotenv";
import { eq, inArray } from "drizzle-orm";
import { getDb } from "./client";
import { savedGroups, users } from "./schema";
import { createGroup, listGroups, updateGroup } from "./queries";

// "Best Couple" x2: saving a group name the user already has must UPDATE that group,
// never create a lookalike. Real Neon DB (the guarantee is a unique constraint, so
// mocking the DB would prove nothing). Skipped when DATABASE_URL isn't set.

// These tests WRITE to a real database (throwaway users, cleaned up after). They read .env.test.local only,
// the Neon test branch, never .env.local (production). Without that file they are skipped.
config({ path: ".env.test.local", quiet: true });
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
    const first = await createGroup(db, A, { groupName: "Best Couple", names: ["Alexis", "Wifey"] });
    const second = await createGroup(db, A, { groupName: "Best Couple", names: ["Alexis", "Wifey", "Sam"] });
    expect(second.id).toBe(first.id);
    expect(second.names).toEqual(["Alexis", "Wifey", "Sam"]);
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

  it("edits a group's name and members", async () => {
    const g = await createGroup(db, A, { groupName: "Edit Me", names: ["A", "B"] });
    const r = await updateGroup(db, A, g.id, { groupName: "Edited", names: ["A", "B", "C"] });
    expect(r.ok && r.group.groupName).toBe("Edited");
    expect(r.ok && r.group.names).toEqual(["A", "B", "C"]);
    expect(r.ok && r.group.id).toBe(g.id);
  });

  it("renaming to a different spelling of its own name is fine", async () => {
    const g = (await listGroups(db, A)).find((x) => x.groupName === "Edited")!;
    const r = await updateGroup(db, A, g.id, { groupName: "  EDITED ", names: g.names });
    expect(r.ok).toBe(true);
  });

  it("renaming onto another of your groups says the name is taken and changes nothing", async () => {
    const edited = (await listGroups(db, A)).find((x) => x.groupName.trim().toLowerCase() === "edited")!;
    const r = await updateGroup(db, A, edited.id, { groupName: "best couple", names: ["Q"] });
    expect(r).toEqual({ ok: false, reason: "name-taken" });
    const after = (await listGroups(db, A)).find((x) => x.id === edited.id)!;
    expect(after.names).toEqual(["A", "B", "C"]);
  });

  it("another user's group can't be edited through your id", async () => {
    const theirs = (await listGroups(db, B))[0];
    const r = await updateGroup(db, A, theirs.id, { groupName: "Hijacked", names: ["Z"] });
    expect(r).toEqual({ ok: false, reason: "not-found" });
    expect((await listGroups(db, B)).find((x) => x.id === theirs.id)!.groupName).toBe(theirs.groupName);
  });

  it("a group includes me by default, and can be saved without me", async () => {
    const withMe = await createGroup(db, A, { groupName: "Me Default", names: ["P"] });
    expect(withMe.includesMe).toBe(true);
    const others = await createGroup(db, A, { groupName: "Others Only", names: ["P", "Q"], includesMe: false });
    expect(others.includesMe).toBe(false);
    expect((await listGroups(db, A)).find((g) => g.id === others.id)?.includesMe).toBe(false);
  });

  it("editing a group can flip includes-me, and saving by the same name keeps what was chosen", async () => {
    const g = (await listGroups(db, A)).find((x) => x.groupName === "Others Only")!;
    const r = await updateGroup(db, A, g.id, { groupName: "Others Only", names: ["P", "Q"], includesMe: true });
    expect(r.ok && r.group.includesMe).toBe(true);
    // an update that doesn't mention includesMe leaves it alone
    const r2 = await updateGroup(db, A, g.id, { groupName: "Others Only", names: ["P"] });
    expect(r2.ok && r2.group.includesMe).toBe(true);
    // re-saving the same name through createGroup (the Start Scanning path) takes the new value
    const again = await createGroup(db, A, { groupName: " others only ", names: ["P"], includesMe: false });
    expect(again.id).toBe(g.id);
    expect(again.includesMe).toBe(false);
  });

  it("the database itself refuses a lookalike, even if the app code were bypassed", async () => {
    await expect(
      db.insert(savedGroups).values({ userId: A, groupName: "BEST COUPLE", names: ["Z"] }),
    ).rejects.toThrow();
  });
});
