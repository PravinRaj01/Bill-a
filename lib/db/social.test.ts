import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { config } from "dotenv";
import { eq, inArray } from "drizzle-orm";
import { getDb } from "./client";
import { billHistory, billShares, friendships, people, paymentClaims, users } from "./schema";
import { listPeople, upsertBill, upsertPerson } from "./queries";
import {
  answerRequest,
  confirmSatisfiedClaims,
  createClaim,
  findUserByUsername,
  getSharedBill,
  listFriendsState,
  listPendingClaimsForOwner,
  listSharablePastBills,
  listSharedBills,
  rejectClaim,
  relationTo,
  removeFriend,
  requestFriend,
  setShareHidden,
  sharePastBills,
} from "./social";

// Friends, shared bills and claims on the real Neon test branch (never .env.local), including the cross-user
// attacks: a stranger can't read a shared bill, claim on it, or answer a request that wasn't sent to them.
config({ path: ".env.test.local", quiet: true });
const hasDb = !!process.env.DATABASE_URL;

const settlement = (paid = {}) => ({ me: "Alexis", paidBy: [{ name: "Alexis", amount: 6000 }], paid });
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
    settlement: settlement(paid),
  },
});

describe.skipIf(!hasDb)("friends, shared bills and claims (real Neon DB)", { timeout: 120_000 }, () => {
  const db = getDb();
  const tag = crypto.randomUUID().slice(0, 8);
  let A = ""; // owner, "Alexis"
  let B = ""; // friend, "Riley"
  let C = ""; // stranger
  let D = ""; // another friend

  beforeAll(async () => {
    const rows = await db
      .insert(users)
      .values([
        { email: `soc-a-${tag}@example.invalid`, name: "Alexis", username: `alexis_${tag}` },
        { email: `soc-b-${tag}@example.invalid`, name: "Riley", username: `riley_${tag}` },
        { email: `soc-c-${tag}@example.invalid`, name: "Casey", username: `casey_${tag}` },
        { email: `soc-d-${tag}@example.invalid`, name: "Sam", username: `sam_${tag}` },
      ])
      .returning({ id: users.id, name: users.name });
    A = rows.find((r) => r.name === "Alexis")!.id;
    B = rows.find((r) => r.name === "Riley")!.id;
    C = rows.find((r) => r.name === "Casey")!.id;
    D = rows.find((r) => r.name === "Sam")!.id;
  });

  afterAll(async () => {
    await db.delete(users).where(inArray(users.id, [A, B, C, D].filter(Boolean)));
  });

  it("finds a user by exact username only, and exposes nothing private", async () => {
    const found = await findUserByUsername(db, `RILEY_${tag}`.toUpperCase());
    expect(found?.id).toBe(B);
    expect(Object.keys(found!).sort()).toEqual(["avatarSeed", "id", "name", "username"]);
    expect(await findUserByUsername(db, `riley_${tag.slice(0, 4)}`)).toBeNull(); // no partial match
    expect(await findUserByUsername(db, "nobody_here_x")).toBeNull();
  });

  it("requests: not yourself, not twice, answered only by the person asked", async () => {
    expect(await requestFriend(db, A, A)).toEqual({ ok: false, reason: "self" });
    expect(await requestFriend(db, A, crypto.randomUUID())).toEqual({ ok: false, reason: "not-found" });
    expect(await requestFriend(db, A, B)).toEqual({ ok: true, status: "requested" });
    expect(await requestFriend(db, A, B)).toEqual({ ok: false, reason: "already-requested" });
    expect(await relationTo(db, A, B)).toBe("outgoing");
    expect(await relationTo(db, B, A)).toBe("incoming");
    expect((await listFriendsState(db, B)).incoming.map((u) => u.id)).toEqual([A]);
    expect((await listFriendsState(db, A)).outgoing.map((u) => u.id)).toEqual([B]);
    expect((await listFriendsState(db, C)).incoming).toEqual([]);

    // the asker can't accept their own request, a stranger can't answer one that wasn't sent to them
    expect(await answerRequest(db, A, B, true)).toBe(false);
    expect(await answerRequest(db, C, A, true)).toBe(false);
    expect(await relationTo(db, A, B)).toBe("outgoing");

    expect(await answerRequest(db, B, A, true)).toBe(true);
    expect(await relationTo(db, A, B)).toBe("friends");
    expect(await requestFriend(db, B, A)).toEqual({ ok: false, reason: "already-friends" });
  });

  it("accepting links a contact on both sides, reusing an existing contact with that name", async () => {
    const pa = await listPeople(db, A);
    const pb = await listPeople(db, B);
    expect(pa.find((p) => p.linkedUserId === B)?.name).toBe("Riley");
    expect(pb.find((p) => p.linkedUserId === A)?.name).toBe("Alexis");

    // D asks A; A already has a plain contact "Sam": it is linked, not duplicated
    await upsertPerson(db, A, { name: "Sam", phone: "+60123456789" });
    expect(await requestFriend(db, D, A)).toEqual({ ok: true, status: "requested" });
    expect(await answerRequest(db, A, D, true)).toBe(true);
    const sam = (await listPeople(db, A)).filter((p) => p.nameKey === "sam");
    expect(sam).toHaveLength(1);
    expect(sam[0].linkedUserId).toBe(D);
    expect(sam[0].phone).toBe("+60123456789"); // the number is kept
  });

  it("a crossing request is accepted, and a declined one stays silent", async () => {
    expect(await requestFriend(db, A, C)).toEqual({ ok: true, status: "requested" });
    expect(await requestFriend(db, C, A)).toEqual({ ok: true, status: "accepted" }); // they asked me first: a yes
    expect(await relationTo(db, A, C)).toBe("friends");
    await removeFriend(db, A, C);
    expect(await relationTo(db, A, C)).toBe("none");

    expect(await requestFriend(db, C, A)).toEqual({ ok: true, status: "requested" });
    expect(await answerRequest(db, A, C, false)).toBe(true); // A declines
    expect(await relationTo(db, C, A)).toBe("none");
    expect(await requestFriend(db, C, A)).toEqual({ ok: true, status: "requested" }); // looks the same to C...
    expect((await listFriendsState(db, A)).incoming).toEqual([]); // ...and A is not asked again
    expect(await requestFriend(db, A, C)).toEqual({ ok: true, status: "requested" }); // A may ask back
    expect(await answerRequest(db, C, A, true)).toBe(true);
    await removeFriend(db, A, C);
  });

  it("a bill naming a linked friend is shared with them, read-only; strangers see nothing", async () => {
    const saved = await upsertBill(db, A, billInput("Dinner", ["Alexis", "Riley"]));
    const list = await listSharedBills(db, B);
    expect(list.map((b) => b.id)).toEqual([saved.id]);
    expect(list[0].asName).toBe("Riley");
    expect(list[0].owner.id).toBe(A);

    const detail = await getSharedBill(db, B, saved.id);
    expect(detail?.data.split.map((s) => s.name)).toEqual(["Alexis", "Riley"]);
    expect(Object.keys(detail!)).not.toContain("clientId"); // can't be saved back as the friend's own bill
    expect(await getSharedBill(db, C, saved.id)).toBeNull();
    expect(await getSharedBill(db, A, saved.id)).toBeNull(); // the owner uses the normal bill view
    expect(await listSharedBills(db, C)).toEqual([]);

    // the owner's own contact who is not a friend ("Casey" unlinked) gets nothing
    const withCasey = await upsertBill(db, A, billInput("Lunch", ["Alexis", "Casey"]));
    expect(await getSharedBill(db, C, withCasey.id)).toBeNull();
    expect(await db.select().from(billShares).where(eq(billShares.billId, withCasey.id))).toEqual([]);
  });

  it("hiding is per person; removing the friend from the bill revokes the share; re-saving doesn't re-add a declined one", async () => {
    const input = billInput("Brunch", ["Alexis", "Riley"]);
    const saved = await upsertBill(db, A, input);
    expect(await setShareHidden(db, B, saved.id, true)).toBe(true);
    expect((await listSharedBills(db, B)).map((b) => b.id)).not.toContain(saved.id);
    expect((await listSharedBills(db, B, { withHidden: true })).map((b) => b.id)).toContain(saved.id);
    expect(await setShareHidden(db, C, saved.id, true)).toBe(false); // someone it isn't shared with can't touch it
    await setShareHidden(db, B, saved.id, false);

    // the owner takes Riley off the bill: the share is gone
    await upsertBill(db, A, { ...input, data: { ...input.data, split: [{ name: "Alexis", amount: 6000, items: "x" }], people: ["Alexis"] } });
    expect(await getSharedBill(db, B, saved.id)).toBeNull();
    // and put back, he is shared again (his name is new to the bill)
    await upsertBill(db, A, input);
    expect(await getSharedBill(db, B, saved.id)).not.toBeNull();
  });

  it("past bills: offered only for that friend's name, only the chosen ones are shared, and a re-save keeps the decision", async () => {
    // saved while D was still a plain name? D is linked now, so make bills that were never shared by writing rows directly
    const mk = async (title: string, names: string[]) => {
      const i = billInput(title, names);
      const [row] = await db.insert(billHistory).values({ userId: A, clientId: i.clientId, billTitle: title, totalAmount: 6000, currency: "RM", data: i.data }).returning({ id: billHistory.id });
      return { id: row.id, input: i };
    };
    const one = await mk("Old one", ["Alexis", "Sam"]);
    const two = await mk("Old two", ["Alexis", "sam"]);
    await mk("Not Sam", ["Alexis", "Riley"]);
    const offered = await listSharablePastBills(db, A, D);
    expect(offered.map((b) => b.billTitle).sort()).toEqual(["Old one", "Old two"]);
    expect(await listSharablePastBills(db, A, C)).toEqual([]); // not a friend
    expect(await listSharablePastBills(db, C, D)).toEqual([]); // C doesn't own those

    // a foreign bill id and one that doesn't name Sam are ignored
    const foreign = await db.insert(billHistory).values({ userId: B, clientId: crypto.randomUUID(), billTitle: "B's", totalAmount: 1, currency: "RM", data: billInput("x", ["Riley", "Sam"]).data }).returning({ id: billHistory.id });
    expect(await sharePastBills(db, A, D, [one.id, foreign[0].id])).toBe(1);
    expect((await listSharedBills(db, D)).map((b) => b.id)).toEqual([one.id]);

    // "Old two" was declined; a later re-save (a paid mark) must not share it
    await upsertBill(db, A, { ...two.input, data: { ...two.input.data, settlement: settlement({ "Sam->Alexis": { at: "2026-01-01", amount: 3000 } }) } });
    expect((await listSharedBills(db, D)).map((b) => b.id)).toEqual([one.id]);
  });

  it("past bills: a bill naming the friend by first name or @username is offered, shared under that name, and stays shared on re-save", async () => {
    await db.update(users).set({ name: "Samuel Tan" }).where(eq(users.id, D)); // contact stays "Sam"
    const mk = async (title: string, names: string[]) => {
      const i = billInput(title, names);
      const [row] = await db.insert(billHistory).values({ userId: A, clientId: i.clientId, billTitle: title, totalAmount: 6000, currency: "RM", data: i.data }).returning({ id: billHistory.id });
      return { id: row.id, input: i };
    };
    const first = await mk("First name", ["Alexis", "Samuel"]);
    const handle = await mk("Handle", ["Alexis", `sam_${tag}`]);
    const offered = await listSharablePastBills(db, A, D);
    expect(offered.filter((b) => [first.id, handle.id].includes(b.id)).map((b) => b.asName).sort()).toEqual(["Samuel", `sam_${tag}`].sort());

    expect(await sharePastBills(db, A, D, [first.id])).toBe(1);
    expect((await getSharedBill(db, D, first.id))?.asName).toBe("Samuel");
    // "Samuel" isn't the contact's name, but re-saving the bill must not drop the share
    await upsertBill(db, A, { ...first.input, data: { ...first.input.data, settlement: settlement({ "Samuel->Alexis": { at: "2026-01-01", amount: 3000 } }) } });
    expect(await getSharedBill(db, D, first.id)).not.toBeNull();
    await db.update(users).set({ name: "Sam" }).where(eq(users.id, D));
  });

  it("claims: only the friend, only a real unpaid transfer to the owner, amount from the server", async () => {
    const saved = await upsertBill(db, A, billInput("Claimable", ["Alexis", "Riley"]));
    expect(await createClaim(db, C, saved.id, "Riley->Alexis")).toEqual({ ok: false, reason: "not-shared" });
    expect(await createClaim(db, B, saved.id, "Alexis->Riley")).toEqual({ ok: false, reason: "no-such-transfer" }); // not mine to pay
    expect(await createClaim(db, B, saved.id, "Riley->Nobody")).toEqual({ ok: false, reason: "no-such-transfer" });
    const r = await createClaim(db, B, saved.id, "Riley->Alexis");
    expect(r.ok && r.claim.amount).toBe(3000);
    expect(r.ok && r.claim.status).toBe("pending");

    const waiting = await listPendingClaimsForOwner(db, A, saved.id);
    expect(waiting).toHaveLength(1);
    expect(waiting[0].from.id).toBe(B);
    expect(await listPendingClaimsForOwner(db, C)).toEqual([]); // not their bills
    expect(await listPendingClaimsForOwner(db, B)).toEqual([]);

    // a stranger cannot reject it; the owner can, and the friend can claim again
    const claimId = waiting[0].id;
    expect(await rejectClaim(db, C, claimId)).toBe(false);
    expect(await rejectClaim(db, B, claimId)).toBe(false); // the friend can't answer their own claim either
    expect(await rejectClaim(db, A, claimId)).toBe(true);
    expect(await listPendingClaimsForOwner(db, A, saved.id)).toEqual([]);
    const again = await createClaim(db, B, saved.id, "Riley->Alexis");
    expect(again.ok && again.claim.status).toBe("pending");

    // the owner marks it paid (through a normal save): the claim confirms itself
    await upsertBill(db, A, { ...billInput("Claimable", ["Alexis", "Riley"], { "Riley->Alexis": { at: "2026-01-01", amount: 3000 } }), clientId: (await db.select({ c: billHistory.clientId }).from(billHistory).where(eq(billHistory.id, saved.id)))[0].c });
    const [claim] = await db.select().from(paymentClaims).where(eq(paymentClaims.billId, saved.id));
    expect(claim.status).toBe("confirmed");
    expect(await createClaim(db, B, saved.id, "Riley->Alexis")).toEqual({ ok: false, reason: "already-paid" });
    await confirmSatisfiedClaims(db, saved.id, undefined); // no-op safety
  });

  it("unfriending ends every share between the two and unlinks the contacts", async () => {
    const saved = await upsertBill(db, A, billInput("Before unfriend", ["Alexis", "Riley"]));
    expect(await getSharedBill(db, B, saved.id)).not.toBeNull();
    const theirs = await db.insert(billHistory).values({ userId: B, clientId: crypto.randomUUID(), billTitle: "B's bill", totalAmount: 1, currency: "RM", data: billInput("x", ["Riley"]).data }).returning({ id: billHistory.id });
    await db.insert(billShares).values({ billId: theirs[0].id, userId: A, asName: "Alexis" });

    await removeFriend(db, A, B);
    expect(await relationTo(db, A, B)).toBe("none");
    expect(await getSharedBill(db, B, saved.id)).toBeNull();
    expect(await getSharedBill(db, A, theirs[0].id)).toBeNull();
    expect((await listPeople(db, A)).find((p) => p.nameKey === "riley")?.linkedUserId).toBeNull();
    expect((await listPeople(db, B)).find((p) => p.nameKey === "alexis")?.linkedUserId).toBeNull();

    // a re-saved bill no longer shares with an unlinked name
    await upsertBill(db, A, { ...billInput("After unfriend", ["Alexis", "Riley"]) });
    expect(await listSharedBills(db, B)).toEqual([]);
    expect(await db.select().from(people).where(eq(people.linkedUserId, B))).toEqual([]);
  });
});
