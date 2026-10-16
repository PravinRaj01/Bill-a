import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { config } from "dotenv";
import { eq, inArray } from "drizzle-orm";
import { getDb } from "./client";
import { billHistory, savedGroups, users } from "./schema";
import {
  billStats,
  billTotalsByCurrency,
  clearBills,
  deleteBills,
  deleteGroup,
  getBill,
  getGroup,
  listBills,
  listGroups,
  listTrackedBills,
  updateGroupNames,
  upsertBill,
  type BillInput,
} from "./queries";
import type { BillData } from "@/types/domain";

// CROSS-USER AUTHORIZATION TEST — the most important test in Phase 2.
//
// Runs against the real Neon database (there is no RLS to fall back on, so
// mocking the DB here would prove nothing). It creates two throwaway users,
// gives user B data, then tries to read / change / delete it AS user A through
// every query function. Each attack must fail or be a no-op, and B's data must
// be byte-for-byte unchanged afterwards. Skipped when no test database is configured.

// These tests WRITE to a real database (throwaway users, cleaned up after). They read .env.test.local only,
// the Neon test branch, never .env.local (production). Without that file they are skipped.
config({ path: ".env.test.local", quiet: true });
const hasDb = !!process.env.DATABASE_URL;

const data: BillData = {
  split: [{ name: "Alexis", amount: 1000, items: "x" }],
  items: { items: [], tax: 0, total: 1000, currency: "RM" },
  people: ["Alexis"],
  reasoning: "test",
};
const bill = (clientId: string, title: string): BillInput => ({
  clientId,
  billTitle: title,
  totalAmount: 1000,
  currency: "RM",
  data,
});

describe.skipIf(!hasDb)("cross-user authorization (real Neon DB)", { timeout: 90_000 }, () => {
  const db = getDb();
  const tag = crypto.randomUUID().slice(0, 8);
  let A = "";
  let B = "";
  let bBillId = "";
  let bGroupId = "";
  const sharedClientId = crypto.randomUUID();

  beforeAll(async () => {
    const [a, b] = await db
      .insert(users)
      .values([
        { email: `authz-a-${tag}@example.invalid`, name: "A" },
        { email: `authz-b-${tag}@example.invalid`, name: "B" },
      ])
      .returning({ id: users.id });
    A = a.id;
    B = b.id;

    const rb = await upsertBill(db, B, bill(sharedClientId, "B secret bill"));
    bBillId = rb.id;
    await upsertBill(db, A, bill(crypto.randomUUID(), "A own bill"));

    const [g] = await db
      .insert(savedGroups)
      .values({ userId: B, groupName: "B secret group", names: ["Alice", "Bob"] })
      .returning({ id: savedGroups.id });
    bGroupId = g.id;
    await db.insert(savedGroups).values({ userId: A, groupName: "A group", names: ["Zed"] });
  });

  afterAll(async () => {
    // FK cascade removes their bills, groups and accounts too.
    if (A && B) await db.delete(users).where(inArray(users.id, [A, B]));
  });

  const bRow = async () => (await db.select().from(billHistory).where(eq(billHistory.id, bBillId)))[0];
  const bGroup = async () => (await db.select().from(savedGroups).where(eq(savedGroups.id, bGroupId)))[0];

  // ---------------- reads ----------------

  it("control: B can read their own bill and group (proves the data really is there)", async () => {
    expect((await getBill(db, B, bBillId))?.billTitle).toBe("B secret bill");
    expect((await getGroup(db, B, bGroupId))?.groupName).toBe("B secret group");
  });

  it("listBills as A never includes B's bills — the old bare select('*') hole", async () => {
    const list = await listBills(db, A);
    expect(list.length).toBeGreaterThan(0);
    expect(list.every((r) => r.billTitle !== "B secret bill")).toBe(true);
    expect(list.some((r) => r.id === bBillId)).toBe(false);
  });

  it("getBill as A on B's id returns null — the old read-by-id hole", async () => {
    expect(await getBill(db, A, bBillId)).toBeNull();
  });

  it("listGroups / getGroup as A never expose B's group", async () => {
    expect((await listGroups(db, A)).some((g) => g.id === bGroupId)).toBe(false);
    expect(await getGroup(db, A, bGroupId)).toBeNull();
  });

  it("an unknown user id sees nothing at all", async () => {
    const ghost = crypto.randomUUID();
    expect(await listBills(db, ghost)).toEqual([]);
    expect(await listGroups(db, ghost)).toEqual([]);
    expect(await getBill(db, ghost, bBillId)).toBeNull();
  });

  // ---------------- writes / deletes ----------------

  it("updateGroupNames as A on B's group changes nothing — the old update-by-id hole", async () => {
    const before = await bGroup();
    expect(await updateGroupNames(db, A, bGroupId, ["HACKED"])).toBeNull();
    expect((await bGroup()).names).toEqual(before.names);
  });

  it("deleteGroup as A on B's group deletes nothing — the old delete-by-id hole", async () => {
    expect(await deleteGroup(db, A, bGroupId)).toBe(false);
    expect(await bGroup()).toBeDefined();
  });

  it("deleteBills as A on B's id deletes nothing — the old bulk-delete hole", async () => {
    expect(await deleteBills(db, A, [bBillId])).toBe(0);
    expect(await bRow()).toBeDefined();
  });

  it("deleteBills with a MIX of own and foreign ids deletes only the own one", async () => {
    const own = await upsertBill(db, A, bill(crypto.randomUUID(), "A temp"));
    expect(await deleteBills(db, A, [own.id, bBillId])).toBe(1);
    expect(await getBill(db, A, own.id)).toBeNull();
    expect(await bRow()).toBeDefined();
  });

  it("deleteBills with an empty list is a safe no-op", async () => {
    expect(await deleteBills(db, A, [])).toBe(0);
  });

  it("upsertBill as A reusing B's client_id creates A's OWN row and never touches B's", async () => {
    const before = await bRow();
    const mine = await upsertBill(db, A, bill(sharedClientId, "A took B's client id"));
    expect(mine.id).not.toBe(bBillId);
    const after = await bRow();
    expect(after.billTitle).toBe(before.billTitle);
    expect(after.userId).toBe(B);
    expect((await getBill(db, A, mine.id))?.billTitle).toBe("A took B's client id");
  });

  it("upsertBill is idempotent per user: same client_id twice = one row, updated", async () => {
    const cid = crypto.randomUUID();
    const first = await upsertBill(db, A, bill(cid, "v1"));
    const second = await upsertBill(db, A, bill(cid, "v2"));
    expect(second.id).toBe(first.id);
    expect((await getBill(db, A, first.id))?.billTitle).toBe("v2");
    const all = (await listBills(db, A)).filter((r) => r.id === first.id);
    expect(all).toHaveLength(1);
  });

  it("billStats counts only the caller's bills, in integer cents", async () => {
    const s = await billStats(db, B);
    expect(s.count).toBe(1);
    expect(s.totalCents).toBe(1000);
    expect(Number.isInteger(s.totalCents)).toBe(true);
  });

  it("billTotalsByCurrency groups the caller's bills by currency and never mixes users", async () => {
    const totals = await billTotalsByCurrency(db, B);
    expect(totals).toEqual([{ currency: "RM", cents: 1000 }]);
  });

  // Destructive to A's data — must stay last.
  it("clearBills as A wipes A's history and leaves B's completely intact", async () => {
    const before = await bRow();
    expect(await clearBills(db, A)).toBeGreaterThan(0);
    expect(await listBills(db, A)).toEqual([]);
    expect(await bRow()).toEqual(before);
    expect((await listBills(db, B)).length).toBe(1);
  });
});

// The History list reads two small slices of the saved jsonb. Through the Neon HTTP driver they must
// come back as real objects (not strings), and a bill saved before settle-up existed must give null.
describe.skipIf(!hasDb)("listBills settle-up slices (real Neon DB)", { timeout: 60_000 }, () => {
  const db = getDb();
  let uid = "";

  beforeAll(async () => {
    const [u] = await db.insert(users).values({ email: `settle-slices-${crypto.randomUUID().slice(0, 8)}@example.invalid`, name: "S" }).returning({ id: users.id });
    uid = u.id;
  });
  afterAll(async () => {
    if (uid) await db.delete(users).where(eq(users.id, uid));
  });

  it("returns the split and settlement as objects, and null settlement for an older bill", async () => {
    const settlement = { me: "Alexis", paidBy: [{ name: "Alexis", amount: 2000 }], paid: { "Riley->Alexis": { at: "2026-10-16", amount: 1000 } }, dueDate: "2026-10-23" };
    const two: BillData = {
      ...data,
      split: [{ name: "Alexis", amount: 1000, items: "x" }, { name: "Riley", amount: 1000, items: "y" }],
      people: ["Alexis", "Riley"],
      settlement,
    };
    await upsertBill(db, uid, { ...bill(crypto.randomUUID(), "Tracked"), totalAmount: 2000, data: two });
    await upsertBill(db, uid, bill(crypto.randomUUID(), "Old bill"));

    const list = await listBills(db, uid);
    const tracked = list.find((r) => r.billTitle === "Tracked")!;
    expect(tracked.settlement).toEqual(settlement);
    expect(tracked.split).toEqual(two.split);
    const old = list.find((r) => r.billTitle === "Old bill")!;
    expect(old.settlement).toBeNull();
    expect(Array.isArray(old.split)).toBe(true);
    // and the list rows still carry no full `data` blob
    expect("data" in tracked).toBe(false);
  });
});

// Balances read only the tracked bills, with just the two slices they need.
describe.skipIf(!hasDb)("listTrackedBills (real Neon DB)", { timeout: 60_000 }, () => {
  const db = getDb();
  let uid = "";
  let otherUid = "";

  beforeAll(async () => {
    const tag = crypto.randomUUID().slice(0, 8);
    const [a, b] = await db
      .insert(users)
      .values([
        { email: `tracked-a-${tag}@example.invalid`, name: "T" },
        { email: `tracked-b-${tag}@example.invalid`, name: "U" },
      ])
      .returning({ id: users.id });
    uid = a.id;
    otherUid = b.id;
  });
  afterAll(async () => {
    if (uid && otherUid) await db.delete(users).where(inArray(users.id, [uid, otherUid]));
  });

  it("returns only tracked bills, newest first, scoped to the user, without the whole bill", async () => {
    const settlement = { me: "Alexis", paidBy: [{ name: "Alexis", amount: 2000 }], paid: {} };
    const tracked: BillData = { ...data, split: [{ name: "Alexis", amount: 1000, items: "x" }, { name: "Riley", amount: 1000, items: "y" }], people: ["Alexis", "Riley"], settlement };
    await upsertBill(db, uid, { ...bill(crypto.randomUUID(), "Tracked one"), totalAmount: 2000, data: tracked });
    await upsertBill(db, uid, bill(crypto.randomUUID(), "Untracked"));
    await upsertBill(db, otherUid, { ...bill(crypto.randomUUID(), "Someone else's tracked"), data: tracked });

    const rows = await listTrackedBills(db, uid);
    expect(rows.map((r) => r.billTitle)).toEqual(["Tracked one"]);
    expect(rows[0].settlement).toEqual(settlement);
    expect(rows[0].split).toHaveLength(2);
    expect("data" in rows[0]).toBe(false);
  });
});
