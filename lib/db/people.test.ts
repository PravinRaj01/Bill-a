import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { config } from "dotenv";
import { eq, inArray, sql } from "drizzle-orm";
import { getDb } from "./client";
import { billHistory, people, savedGroups, users } from "./schema";
import { deletePerson, listPeople, suggestPeople, updatePerson, upsertPerson } from "./queries";

// Contacts are keyed by the session's user id only. Real Neon DB (the test branch, never .env.local); skipped
// without .env.test.local. Throwaway users are removed afterwards (their contacts, bills and groups cascade).
config({ path: ".env.test.local", quiet: true });
const hasDb = !!process.env.DATABASE_URL;

const bill = (userId: string, title: string, names: string[]) => ({
  userId,
  clientId: crypto.randomUUID(),
  billTitle: title,
  totalAmount: 1000,
  currency: "RM",
  data: { split: names.map((name) => ({ name, amount: 100, items: "x" })), items: { items: [], tax: 0, total: 1000, currency: "RM" }, people: names, reasoning: "t" },
});

describe.skipIf(!hasDb)("people (real Neon DB)", { timeout: 90_000 }, () => {
  const db = getDb();
  const tag = crypto.randomUUID().slice(0, 8);
  let A = "";
  let B = "";

  beforeAll(async () => {
    const [a, b] = await db
      .insert(users)
      .values([
        { email: `ppl-a-${tag}@example.invalid`, name: "A" },
        { email: `ppl-b-${tag}@example.invalid`, name: "B" },
      ])
      .returning({ id: users.id });
    A = a.id;
    B = b.id;
  });

  afterAll(async () => {
    await db.delete(users).where(inArray(users.id, [A, B].filter(Boolean)));
  });

  it("adds a contact, and the same name in another case updates it instead of duplicating", async () => {
    const first = await upsertPerson(db, A, { name: "Riley", phone: "+60123456789" });
    const again = await upsertPerson(db, A, { name: "  RILEY ", phone: null });
    expect(again.id).toBe(first.id);
    expect(again.name).toBe("Riley"); // keeps the original spelling
    expect(again.phone).toBe("+60123456789"); // a missing number never wipes the saved one
    expect((await listPeople(db, A)).filter((p) => p.nameKey === "riley")).toHaveLength(1);
  });

  it("two adds of the same name at the same instant still make one contact", async () => {
    const results = await Promise.all([upsertPerson(db, A, { name: "Casey" }), upsertPerson(db, A, { name: "casey" })]);
    expect(new Set(results.map((r) => r.id)).size).toBe(1);
  });

  it("is scoped to the user: another user has their own, and can't see, edit or delete mine", async () => {
    const mine = (await listPeople(db, A)).find((p) => p.nameKey === "riley")!;
    const theirs = await upsertPerson(db, B, { name: "Riley" });
    expect(theirs.id).not.toBe(mine.id);
    expect((await listPeople(db, B)).map((p) => p.id)).toEqual([theirs.id]);
    expect(await updatePerson(db, B, mine.id, { name: "Hijacked" })).toEqual({ ok: false, reason: "not-found" });
    expect(await deletePerson(db, B, mine.id)).toBe(false);
    const after = (await listPeople(db, A)).find((p) => p.id === mine.id)!;
    expect(after.name).toBe("Riley");
    expect(after.phone).toBe("+60123456789");
  });

  it("edits a number and a name; null clears the number; renaming onto another contact is 'name-taken'", async () => {
    const riley = (await listPeople(db, A)).find((p) => p.nameKey === "riley")!;
    const cleared = await updatePerson(db, A, riley.id, { phone: null });
    expect(cleared.ok && cleared.person.phone).toBeNull();
    const renamed = await updatePerson(db, A, riley.id, { name: "Riley T", phone: "+6598765432" });
    expect(renamed.ok && renamed.person.name).toBe("Riley T");
    expect(await updatePerson(db, A, riley.id, { name: "casey" })).toEqual({ ok: false, reason: "name-taken" });
    expect((await listPeople(db, A)).find((p) => p.id === riley.id)?.name).toBe("Riley T");
  });

  it("the database itself refuses a malformed phone number", async () => {
    await expect(db.insert(people).values({ userId: A, name: "Bad", phone: "012-345" })).rejects.toThrow();
    await expect(db.insert(people).values({ userId: A, name: "Bad2", phone: "+0123456789" })).rejects.toThrow();
    await expect(db.execute(sql`update people set phone = '60123456789' where user_id = ${A}`)).rejects.toThrow();
  });

  it("deletes your own contact", async () => {
    const c = await upsertPerson(db, A, { name: "Temp" });
    expect(await deletePerson(db, A, c.id)).toBe(true);
    expect((await listPeople(db, A)).some((p) => p.id === c.id)).toBe(false);
  });

  it("suggests the names from your bills and groups that aren't contacts, most frequent first, spelled as most often written", async () => {
    await db.insert(billHistory).values([
      bill(A, "one", ["Alexis", "Sam", "sam", "Jo"]),
      bill(A, "two", ["Alexis", "Sam", "Jo"]),
      bill(A, "three", ["Alexis", "Sam", "Mei"]),
    ]);
    await db.insert(savedGroups).values({ userId: A, groupName: "Crew", names: ["Mei", "Jo", "Riley T", "Casey"] });
    // someone else's bills must never leak into my suggestions
    await db.insert(billHistory).values(bill(B, "theirs", ["Stranger", "Stranger2"]));
    const s = await suggestPeople(db, A, ["Alexis"]);
    const names = s.map((x) => x.name);
    expect(names).not.toContain("Alexis"); // excluded (it's me)
    expect(names).not.toContain("Riley T"); // already a contact
    expect(names).not.toContain("Casey"); // already a contact
    expect(names).not.toContain("Stranger");
    expect(s[0]).toEqual({ name: "Sam", count: 4 }); // Sam x3 + sam x1, shown as most often written
    expect(s.find((x) => x.name === "Jo")?.count).toBe(3);
    expect(s.find((x) => x.name === "Mei")?.count).toBe(2);
    expect(s.map((x) => x.count)).toEqual([...s.map((x) => x.count)].sort((a, b) => b - a));
    expect((await suggestPeople(db, A, [], 1))).toHaveLength(1);
  });

  it("never suggests you, under any name you were called on your own bills", async () => {
    const own = bill(A, "mine", ["Old Nickname", "Tan", "Quinn"]);
    await db.insert(billHistory).values([
      { ...own, data: { ...own.data, settlement: { me: "Old Nickname", paidBy: [], paid: {} } } },
      { ...bill(A, "hosted-with-me", ["Quinn", "Hana"]), data: { ...bill(A, "x", ["Quinn", "Hana"]).data, participants: [{ name: "Hana", me: true }, { name: "Quinn" }] } },
    ]);
    const names = (await suggestPeople(db, A, [])).map((x) => x.name);
    expect(names).not.toContain("Old Nickname"); // settlement.me
    expect(names).not.toContain("Hana"); // participants me
    expect(names).toContain("Quinn");
    expect(names).toContain("Tan");
  });

  it("a new user with no history gets no suggestions", async () => {
    expect(await suggestPeople(db, B, ["B"])).toEqual([{ name: "Stranger", count: 1 }, { name: "Stranger2", count: 1 }]);
    const empty = await suggestPeople(db, crypto.randomUUID(), []);
    expect(empty).toEqual([]);
  });
});
