import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { config } from "dotenv";
import { and, eq, inArray } from "drizzle-orm";
import { getDb } from "./client";
import { billShares, notifications, paymentClaims, people, users } from "./schema";
import { getBill, listBills, listHiddenBills, listPeople, listTrackedBills, setBillHidden, upsertBill, upsertPerson } from "./queries";
import { welcomeNewUser } from "./notifications";
import { answerRequest, createClaim, findDuplicateBills, linkCandidates, linkContactToFriend, listSharablePastBills, requestFriend } from "./social";
import type { BillData, Participant } from "@/types/domain";

// Bills link friends by ACCOUNT (participants[].userId), so a share survives a rename and a forged id does nothing.
// On the real Neon test branch only (.env.test.local).
config({ path: ".env.test.local", quiet: true });
const hasDb = !!process.env.DATABASE_URL;

describe.skipIf(!hasDb)("friends are linked to bills by account (real Neon DB)", { timeout: 120_000 }, () => {
  const db = hasDb ? getDb() : (null as never);
  const tag = crypto.randomUUID().slice(0, 8);
  let A = ""; // owner "Alexis"
  let B = ""; // friend "Pravin Raj"
  let C = ""; // a stranger

  beforeAll(async () => {
    const rows = await db
      .insert(users)
      .values([
        { email: `lnk-a-${tag}@example.invalid`, name: "Alexis", username: `alexis_l${tag}` },
        { email: `lnk-b-${tag}@example.invalid`, name: "Pravin Raj", username: `pravin_l${tag}` },
        { email: `lnk-c-${tag}@example.invalid`, name: "Stranger", username: `stranger_l${tag}` },
      ])
      .returning({ id: users.id, name: users.name });
    A = rows.find((r) => r.name === "Alexis")!.id;
    B = rows.find((r) => r.name === "Pravin Raj")!.id;
    C = rows.find((r) => r.name === "Stranger")!.id;
    await requestFriend(db, A, B);
    await answerRequest(db, B, A, true);
  });

  afterAll(async () => {
    await db.delete(users).where(inArray(users.id, [A, B, C].filter(Boolean)));
  });

  const bill = (participants: Participant[], paid = {}, total = 6000): BillData => ({
    split: participants.map((p) => ({ name: p.name, amount: total / participants.length, items: "x" })),
    items: { items: [], tax: 0, total, currency: "RM" },
    people: participants.map((p) => p.name),
    reasoning: "t",
    participants,
    settlement: { me: "Alexis", paidBy: [{ name: "Alexis", amount: total }], paid },
  });
  const input = (clientId: string, data: BillData) => ({ clientId, billTitle: "Dinner", totalAmount: 6000, currency: "RM", data });
  const sharesOf = (billId: string) => db.select().from(billShares).where(eq(billShares.billId, billId));

  it("shares with a linked friend under whatever name the bill uses, with no matching contact name", async () => {
    const saved = await upsertBill(db, A, input(crypto.randomUUID(), bill([{ name: "Alexis", me: true }, { name: "Pravin", userId: B }])));
    const shares = await sharesOf(saved.id);
    expect(shares.map((s) => [s.userId, s.asName])).toEqual([[B, "Pravin"]]);
  });

  it("strips a forged id and a non-friend id: no share, nothing left in the stored bill", async () => {
    const saved = await upsertBill(db, A, input(crypto.randomUUID(), bill([{ name: "Alexis", me: true }, { name: "Mallory", userId: C }, { name: "Self", userId: A }])));
    expect(await sharesOf(saved.id)).toHaveLength(0);
    const stored = await getBill(db, A, saved.id);
    expect(stored!.data.participants!.some((p) => p.userId)).toBe(false);
  });

  it("a stale save under a different name keeps the share; dropping the link removes it, adding it back notifies again", async () => {
    const clientId = crypto.randomUUID();
    const first = await upsertBill(db, A, input(clientId, bill([{ name: "Alexis", me: true }, { name: "Pravin Raj", userId: B }])));
    // the bill still carries the old text but the same account: the share stays
    await upsertBill(db, A, input(clientId, bill([{ name: "Alexis", me: true }, { name: "Praveen", userId: B }])));
    expect((await sharesOf(first.id)).map((s) => s.asName)).toEqual(["Praveen"]);
    // unlinked (the owner stops sharing): the share goes, the person stays on the bill as a plain name
    await upsertBill(db, A, input(clientId, bill([{ name: "Alexis", me: true }, { name: "Praveen" }])));
    expect(await sharesOf(first.id)).toHaveLength(0);
    expect((await getBill(db, A, first.id))!.data.people).toContain("Praveen");
    const told = await db.select().from(notifications).where(and(eq(notifications.userId, B), eq(notifications.billId, first.id), eq(notifications.kind, "bill_unshared")));
    expect(told).toHaveLength(1); // the friend hears it left their History
    await db.update(notifications).set({ readAt: new Date() }).where(and(eq(notifications.userId, B), eq(notifications.billId, first.id)));
    await upsertBill(db, A, input(clientId, bill([{ name: "Alexis", me: true }, { name: "Praveen", userId: B }])));
    const unread = await db.select().from(notifications).where(and(eq(notifications.userId, B), eq(notifications.billId, first.id), eq(notifications.kind, "bill_added")));
    expect(unread[0].readAt).toBeNull();
  });

  it("closes a claim the edited bill no longer matches and tells the friend", async () => {
    const clientId = crypto.randomUUID();
    const saved = await upsertBill(db, A, input(clientId, bill([{ name: "Alexis", me: true }, { name: "Pravin Raj", userId: B }])));
    expect(await createClaim(db, B, saved.id, "Pravin Raj->Alexis")).toMatchObject({ ok: true });
    // the total changes, so what Pravin owes changes
    await upsertBill(db, A, { ...input(clientId, bill([{ name: "Alexis", me: true }, { name: "Pravin Raj", userId: B }], {}, 9000)), totalAmount: 9000 });
    const [claim] = await db.select().from(paymentClaims).where(eq(paymentClaims.billId, saved.id));
    expect(claim.status).toBe("rejected");
    const told = await db.select().from(notifications).where(and(eq(notifications.userId, B), eq(notifications.kind, "claim_changed")));
    expect(told.length).toBe(1);
  });

  it("past bills: the friend's first name is not guessed when another pal already goes by it", async () => {
    // A has a contact "Pravin Raj" (linked to B). A bill names "Pravin" and "Pravin Kumar" (a different pal).
    const mine = (await listPeople(db, A)).find((p) => p.linkedUserId === B)!;
    expect(mine.name).toBe("Pravin Raj");
    const one = await upsertBill(db, A, input(crypto.randomUUID(), { ...bill([{ name: "Alexis", me: true }, { name: "Pravin" }]), settlement: undefined }));
    expect((await listSharablePastBills(db, A, B)).some((b) => b.id === one.id)).toBe(true); // nobody else is a Pravin: offered
    await upsertPerson(db, A, { name: "Pravin Kumar" });
    expect((await listSharablePastBills(db, A, B)).some((b) => b.id === one.id)).toBe(false); // now it could be either: not offered
    await db.delete(people).where(and(eq(people.userId, A), eq(people.name, "Pravin Kumar")));
  });

  it("an existing contact can be linked to the friend, keeping its name and number", async () => {
    await upsertPerson(db, A, { name: "Pravin", phone: "+60123456789" });
    const cands = await linkCandidates(db, A, B);
    expect(cands.map((c) => c.name)).toEqual(["Pravin"]);
    expect(await linkContactToFriend(db, A, C, cands[0].id)).toBe(false); // C is not a friend
    expect(await linkContactToFriend(db, A, B, cands[0].id)).toBe(true);
    const mine = await listPeople(db, A);
    const linked = mine.filter((p) => p.linkedUserId === B);
    expect(linked.map((p) => [p.name, p.phone])).toEqual([["Pravin", "+60123456789"]]); // one contact, the old one
    expect(await linkCandidates(db, A, B)).toEqual([]);
  });

  it("a new account gets one welcome notice that points at the AI key guide, however often it is asked for", async () => {
    await welcomeNewUser(db, C);
    await welcomeNewUser(db, C);
    const rows = await db.select().from(notifications).where(and(eq(notifications.userId, C), eq(notifications.kind, "welcome")));
    expect(rows).toHaveLength(1);
  });

  it("the same bill logged by both is flagged; hiding a copy takes it out of History and balances, and only the owner can", async () => {
    // A shares a 77.00 dinner with B; B logged the same 77.00 dinner too.
    await upsertBill(db, A, { clientId: crypto.randomUUID(), billTitle: "Hawker dinner", totalAmount: 7700, currency: "RM", data: bill([{ name: "Alexis", me: true }, { name: "Pravin", userId: B }], {}, 7700) });
    const mine = await upsertBill(db, B, { clientId: crypto.randomUUID(), billTitle: "Dinner with Alexis", totalAmount: 7700, currency: "RM", data: { ...bill([{ name: "Pravin", me: true }, { name: "Alexis" }], {}, 7700), settlement: { me: "Pravin", paidBy: [{ name: "Pravin", amount: 7700 }], paid: {} } } });
    const pairs = await findDuplicateBills(db, B);
    expect(pairs.map((p) => [p.ownId, p.sharedTitle])).toEqual([[mine.id, "Hawker dinner"]]);
    expect(await findDuplicateBills(db, A)).toEqual([]); // A has no second copy

    expect(await setBillHidden(db, A, mine.id, true)).toBe(false); // not A's bill
    expect(await setBillHidden(db, B, mine.id, true)).toBe(true);
    expect((await listBills(db, B)).some((b) => b.id === mine.id)).toBe(false);
    expect((await listTrackedBills(db, B)).some((b) => b.id === mine.id)).toBe(false);
    expect((await listHiddenBills(db, B)).map((b) => b.id)).toEqual([mine.id]);
    expect(await findDuplicateBills(db, B)).toEqual([]); // hidden: no longer a duplicate
    expect(await setBillHidden(db, B, mine.id, false)).toBe(true);
    expect((await listBills(db, B)).some((b) => b.id === mine.id)).toBe(true);
  });

  it("past bills never offer the owner's own bill, even when the owner shares the friend's first name", async () => {
    // Owner "Pravin" (own first name equals the friend's first name) with the friend recorded as "Pravin Raj".
    const own = await upsertBill(db, A, input(crypto.randomUUID(), { ...bill([{ name: "Pravin", me: true }, { name: "Sam" }]), settlement: undefined }));
    const offered = await listSharablePastBills(db, A, B);
    expect(offered.some((b) => b.id === own.id)).toBe(false);
  });
});
