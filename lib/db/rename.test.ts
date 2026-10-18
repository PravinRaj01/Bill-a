import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { config } from "dotenv";
import { eq, inArray } from "drizzle-orm";
import { getDb } from "./client";
import { billShares, paymentClaims, people, savedGroups, users } from "./schema";
import { createGroup, getBill, setUserName, upsertBill } from "./queries";
import { answerRequest, createClaim, requestFriend } from "./social";

// A friend renames themselves: the owner's bills shared with them, the contact and the saved groups follow.
// On the real Neon test branch only (.env.test.local).
config({ path: ".env.test.local", quiet: true });
const hasDb = !!process.env.DATABASE_URL;

describe.skipIf(!hasDb)("a friend's rename carries over (real Neon DB)", { timeout: 120_000 }, () => {
  const db = hasDb ? getDb() : (null as never);
  const tag = crypto.randomUUID().slice(0, 8);
  let A = ""; // owner "Alexis"
  let B = ""; // friend "Pravin Raj"

  beforeAll(async () => {
    const rows = await db
      .insert(users)
      .values([
        { email: `ren-a-${tag}@example.invalid`, name: "Alexis", username: `alexis_r${tag}` },
        { email: `ren-b-${tag}@example.invalid`, name: "Pravin Raj", username: `pravin_r${tag}` },
      ])
      .returning({ id: users.id, name: users.name });
    A = rows.find((r) => r.name === "Alexis")!.id;
    B = rows.find((r) => r.name === "Pravin Raj")!.id;
    await requestFriend(db, A, B);
    await answerRequest(db, B, A, true); // A now has a contact "Pravin Raj" linked to B
  });

  afterAll(async () => {
    await db.delete(users).where(inArray(users.id, [A, B].filter(Boolean)));
  });

  it("rewrites the shared bill, the share, a claim, the contact and a group; leaves nicknames alone", async () => {
    const data = (names: string[], paid = {}) => ({
      split: names.map((name) => ({ name, amount: 6000 / names.length, items: "x" })),
      items: { items: [], tax: 0, total: 6000, currency: "RM" },
      people: names,
      reasoning: "t",
      settlement: { me: "Alexis", paidBy: [{ name: "Alexis", amount: 6000 }], paid },
    });
    const shared = await upsertBill(db, A, { clientId: crypto.randomUUID(), billTitle: "Dinner", totalAmount: 6000, currency: "RM", data: data(["Alexis", "Pravin Raj"]) });
    expect(await createClaim(db, B, shared.id, "Pravin Raj->Alexis")).toMatchObject({ ok: true });
    await db.insert(billShares).values({ billId: (await upsertBill(db, A, { clientId: crypto.randomUUID(), billTitle: "Nick", totalAmount: 6000, currency: "RM", data: data(["Alexis", "PJ"]) })).id, userId: B, asName: "PJ" });
    await createGroup(db, A, { groupName: `Lunch ${tag}`, names: ["Pravin Raj", "Sam"] });

    await setUserName(db, B, "Praveen Kumar");

    const bill = await getBill(db, A, shared.id);
    expect(bill!.data.split.map((s) => s.name)).toEqual(["Alexis", "Praveen Kumar"]);
    const [share] = await db.select().from(billShares).where(eq(billShares.billId, shared.id));
    expect(share.asName).toBe("Praveen Kumar");
    const [claim] = await db.select().from(paymentClaims).where(eq(paymentClaims.billId, shared.id));
    expect(claim.transferKey).toBe("Praveen Kumar->Alexis");
    const [contact] = await db.select().from(people).where(eq(people.linkedUserId, B));
    expect(contact.name).toBe("Praveen Kumar");
    const [group] = await db.select().from(savedGroups).where(eq(savedGroups.userId, A));
    expect(group.names).toEqual(["Praveen Kumar", "Sam"]);
    // the nickname bill is untouched
    const nick = await db.select().from(billShares).where(eq(billShares.asName, "PJ"));
    expect(nick).toHaveLength(1);
  });
});
