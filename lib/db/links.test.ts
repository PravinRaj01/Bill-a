import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { config } from "dotenv";
import { and, eq, inArray } from "drizzle-orm";
import { getDb } from "./client";
import { billShares, notifications, paymentClaims, users } from "./schema";
import { getBill, upsertBill } from "./queries";
import { answerRequest, createClaim, listSharablePastBills, requestFriend } from "./social";
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

  it("past bills never offer the owner's own bill, even when the owner shares the friend's first name", async () => {
    // Owner "Pravin" (own first name equals the friend's first name) with the friend recorded as "Pravin Raj".
    const own = await upsertBill(db, A, input(crypto.randomUUID(), { ...bill([{ name: "Pravin", me: true }, { name: "Sam" }]), settlement: undefined }));
    const offered = await listSharablePastBills(db, A, B);
    expect(offered.some((b) => b.id === own.id)).toBe(false);
  });
});
