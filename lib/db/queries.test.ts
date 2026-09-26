import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { config } from "dotenv";
import { eq, inArray } from "drizzle-orm";
import { getDb } from "./client";
import { billHistory, savedGroups, users } from "./schema";
import {
  billStats,
  clearBills,
  deleteBills,
  deleteGroup,
  getBill,
  getGroup,
  listBills,
  listGroups,
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
// be byte-for-byte unchanged afterwards. Skipped when DATABASE_URL isn't set.

config({ path: ".env.local" });
const hasDb = !!process.env.DATABASE_URL;

const data: BillData = {
  split: [{ name: "Pravin", amount: 1000, items: "x" }],
  items: { items: [], tax: 0, total: 1000, currency: "RM" },
  people: ["Pravin"],
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

  // Destructive to A's data — must stay last.
  it("clearBills as A wipes A's history and leaves B's completely intact", async () => {
    const before = await bRow();
    expect(await clearBills(db, A)).toBeGreaterThan(0);
    expect(await listBills(db, A)).toEqual([]);
    expect(await bRow()).toEqual(before);
    expect((await listBills(db, B)).length).toBe(1);
  });
});
