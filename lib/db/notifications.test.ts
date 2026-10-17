import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { config } from "dotenv";
import { eq, inArray } from "drizzle-orm";
import { getDb } from "./client";
import { billHistory, notifications, users } from "./schema";
import { deleteBills, upsertBill } from "./queries";
import { listNotifications, markAllRead, markRead, notify, unreadCount } from "./notifications";
import { answerRequest, createClaim, listNudgeable, nudgeSharedFriend, rejectClaim, removeFriend, requestFriend } from "./social";

// The inbox and every event that feeds it, on the real Neon test branch, including the cross-user attacks:
// nobody can list, count or mark read another user's notifications, or nudge on a bill that isn't theirs.
config({ path: ".env.test.local", quiet: true });
const hasDb = !!process.env.DATABASE_URL;

const billInput = (title: string, names: string[], paid = {}) => ({
  clientId: crypto.randomUUID(),
  billTitle: title,
  totalAmount: 6000,
  currency: "RM",
  data: {
    split: names.map((name) => ({ name, amount: 6000 / names.length, items: "x" })),
    items: { items: [], tax: 0, total: 6000, currency: "RM" },
    people: names,
    reasoning: "t",
    settlement: { me: "Alexis", paidBy: [{ name: "Alexis", amount: 6000 }], paid },
  },
});

describe.skipIf(!hasDb)("notifications (real Neon DB)", { timeout: 120_000 }, () => {
  const db = getDb();
  const tag = crypto.randomUUID().slice(0, 8);
  let A = ""; // owner "Alexis"
  let B = ""; // friend "Riley"
  let C = ""; // stranger

  beforeAll(async () => {
    const rows = await db
      .insert(users)
      .values([
        { email: `ntf-a-${tag}@example.invalid`, name: "Alexis", username: `alexis_n${tag}` },
        { email: `ntf-b-${tag}@example.invalid`, name: "Riley", username: `riley_n${tag}` },
        { email: `ntf-c-${tag}@example.invalid`, name: "Casey", username: `casey_n${tag}` },
      ])
      .returning({ id: users.id, name: users.name });
    A = rows.find((r) => r.name === "Alexis")!.id;
    B = rows.find((r) => r.name === "Riley")!.id;
    C = rows.find((r) => r.name === "Casey")!.id;
  });

  afterAll(async () => {
    await db.delete(users).where(inArray(users.id, [A, B, C].filter(Boolean)));
  });

  const kinds = async (u: string) => (await listNotifications(db, u)).map((n) => n.kind);

  it("a friend request and its acceptance reach the right inbox, once", async () => {
    await requestFriend(db, A, B);
    expect(await kinds(B)).toEqual(["friend_request"]);
    expect(await kinds(A)).toEqual([]);
    const [req] = await listNotifications(db, B);
    expect(req.actor?.name).toBe("Alexis");

    await answerRequest(db, B, A, true);
    expect(await kinds(A)).toEqual(["friend_accepted"]);
    expect(await unreadCount(db, B)).toBe(1);
  });

  it("a declined request tells the asker nothing", async () => {
    await requestFriend(db, C, A);
    expect(await kinds(A)).toContain("friend_request");
    const before = (await kinds(C)).length;
    await answerRequest(db, A, C, false);
    expect((await kinds(C)).length).toBe(before);
  });

  it("repeating an event does not notify twice, but a refresh brings it back as unread", async () => {
    const dedupe = `test:${tag}`;
    expect(await notify(db, { userId: B, kind: "nudge", actorId: A, dedupe })).toBe(true);
    expect(await notify(db, { userId: B, kind: "nudge", actorId: A, dedupe })).toBe(false);
    const [row] = await db.select().from(notifications).where(eq(notifications.dedupe, dedupe));
    await markRead(db, B, [row.id]);
    expect(await notify(db, { userId: B, kind: "nudge", actorId: A, dedupe, refresh: true })).toBe(true);
    const [again] = await db.select().from(notifications).where(eq(notifications.dedupe, dedupe));
    expect(again.readAt).toBeNull();
    expect(await notify(db, { userId: A, kind: "nudge", actorId: A, dedupe: `self:${tag}` })).toBe(false); // never yourself
  });

  it("adding a friend to a bill notifies them; re-saving does not repeat it", async () => {
    const input = billInput("Dinner out", ["Alexis", "Riley"]);
    const saved = await upsertBill(db, A, input);
    await upsertBill(db, A, input);
    const added = (await listNotifications(db, B)).filter((n) => n.kind === "bill_added");
    expect(added).toHaveLength(1);
    expect(added[0].billId).toBe(saved.id);
    expect(added[0].payload.title).toBe("Dinner out");
  });

  it("claim made, rejected, then confirmed by the owner's paid mark", async () => {
    const input = billInput("Claims night", ["Alexis", "Riley"]);
    const saved = await upsertBill(db, A, input);
    const claim = await createClaim(db, B, saved.id, "Riley->Alexis");
    expect(claim.ok).toBe(true);
    const ownerSees = (await listNotifications(db, A)).find((n) => n.kind === "claim_made" && n.billId === saved.id);
    expect(ownerSees?.payload.amount).toBe(3000); // the server's amount
    expect(ownerSees?.actor?.name).toBe("Riley");

    if (!claim.ok) return;
    expect(await rejectClaim(db, C, claim.claim.id)).toBe(false); // a stranger can't answer it...
    expect((await listNotifications(db, B)).some((n) => n.kind === "claim_rejected")).toBe(false); // ...nor notify
    expect(await rejectClaim(db, A, claim.claim.id)).toBe(true);
    expect((await listNotifications(db, B)).some((n) => n.kind === "claim_rejected" && n.billId === saved.id)).toBe(true);

    await createClaim(db, B, saved.id, "Riley->Alexis"); // says it again
    await upsertBill(db, A, { ...input, data: { ...input.data, settlement: { ...input.data.settlement, paid: { "Riley->Alexis": { at: "2026-01-01", amount: 3000 } } } } });
    expect((await listNotifications(db, B)).some((n) => n.kind === "claim_confirmed" && n.billId === saved.id)).toBe(true);
  });

  it("nudges: owner only, only while money is owed, once a day per bill", async () => {
    const saved = await upsertBill(db, A, billInput("Nudge me", ["Alexis", "Riley"]));
    const day = "2026-03-01";
    expect((await listNudgeable(db, A, saved.id, day)).map((x) => [x.friend.id, x.owed, x.nudgedToday])).toEqual([[B, 3000, false]]);
    expect(await listNudgeable(db, B, saved.id, day)).toEqual([]); // the friend isn't the owner
    expect(await listNudgeable(db, C, saved.id, day)).toEqual([]);

    expect(await nudgeSharedFriend(db, B, saved.id, A, day)).toEqual({ ok: false, reason: "not-shared" }); // friend can't nudge the owner
    expect(await nudgeSharedFriend(db, C, saved.id, B, day)).toEqual({ ok: false, reason: "not-shared" });
    expect(await nudgeSharedFriend(db, A, saved.id, C, day)).toEqual({ ok: false, reason: "not-shared" }); // not on the bill
    expect(await nudgeSharedFriend(db, A, saved.id, B, day)).toEqual({ ok: true });
    expect(await nudgeSharedFriend(db, A, saved.id, B, day)).toEqual({ ok: false, reason: "already-today" });
    expect(await nudgeSharedFriend(db, A, saved.id, B, "2026-03-02")).toEqual({ ok: true });
    expect((await listNudgeable(db, A, saved.id, day))[0].nudgedToday).toBe(true);

    const nudges = (await listNotifications(db, B)).filter((n) => n.kind === "nudge" && n.billId === saved.id);
    expect(nudges).toHaveLength(2);
    expect(nudges[0].payload).toMatchObject({ title: "Nudge me", amount: 3000, currency: "RM" });
  });

  it("nothing owed, nothing to nudge", async () => {
    const input = billInput("Already square", ["Alexis", "Riley"], { "Riley->Alexis": { at: "2026-01-01", amount: 3000 } });
    const saved = await upsertBill(db, A, input);
    expect(await nudgeSharedFriend(db, A, saved.id, B, "2026-03-05")).toEqual({ ok: false, reason: "nothing-owed" });
  });

  it("deleting a shared bill tells the friend, with the title and no bill to open", async () => {
    const saved = await upsertBill(db, A, billInput("Doomed dinner", ["Alexis", "Riley"]));
    expect(await deleteBills(db, A, [saved.id])).toBe(1);
    const gone = (await listNotifications(db, B)).find((n) => n.kind === "bill_deleted");
    expect(gone?.payload.title).toBe("Doomed dinner");
    expect(gone?.billId).toBeNull();
    expect(gone?.actor?.name).toBe("Alexis");
    // the "added to a bill" message for it survives the delete, just without a bill to open
    expect((await listNotifications(db, B)).find((n) => n.payload.title === "Doomed dinner" && n.kind === "bill_added")?.billId).toBeNull();
    expect(await db.select().from(billHistory).where(eq(billHistory.id, saved.id))).toEqual([]);
  });

  it("a stranger's delete of someone else's bill id notifies nobody", async () => {
    const saved = await upsertBill(db, A, billInput("Not yours", ["Alexis", "Riley"]));
    const before = (await kinds(B)).length;
    expect(await deleteBills(db, C, [saved.id])).toBe(0);
    expect((await kinds(B)).length).toBe(before);
  });

  it("mark read only touches your own notifications", async () => {
    const mine = await listNotifications(db, B);
    const theirs = await listNotifications(db, A);
    expect(await markRead(db, C, mine.map((n) => n.id))).toBe(0); // a stranger can't mark B's as read
    expect(await unreadCount(db, B)).toBeGreaterThan(0);
    expect(await markRead(db, B, [theirs[0].id])).toBe(0); // nor can B mark A's
    expect(await markRead(db, B, [])).toBe(0);
    expect(await unreadCount(db, A)).toBeGreaterThan(0);

    const unread = await unreadCount(db, B);
    expect(await markAllRead(db, B)).toBe(unread);
    expect(await unreadCount(db, B)).toBe(0);
    expect(await unreadCount(db, A)).toBeGreaterThan(0); // A's are untouched
    expect(await markAllRead(db, B)).toBe(0);
    expect(await listNotifications(db, C)).toEqual(expect.not.arrayContaining(mine)); // C never sees B's
  });

  it("unfriending is quiet and leaves the inbox alone", async () => {
    const before = (await kinds(B)).length;
    await removeFriend(db, A, B);
    expect((await kinds(B)).length).toBe(before);
  });
});
