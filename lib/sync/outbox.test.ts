import "fake-indexeddb/auto";
import { IDBFactory } from "fake-indexeddb";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  closeOutbox,
  enqueueBill,
  flushOutbox,
  outboxCounts,
  type PushFn,
} from "./outbox";
import type { BillPayload } from "@/lib/validation/bill";

const bill = (clientId: string, title = "T"): BillPayload => ({
  clientId,
  billTitle: title,
  totalAmount: 1000,
  currency: "RM",
  data: {
    split: [{ name: "A", amount: 1000, items: "x" }],
    items: { items: [], tax: 0, total: 1000, currency: "RM" },
    people: ["A"],
    reasoning: "",
  },
});

// A fake server that behaves like pushOutbox: idempotent on clientId.
function fakeServer() {
  const rows = new Map<string, BillPayload>();
  const calls: string[][] = [];
  const push: PushFn = async (batch) => {
    calls.push(batch.map((b) => b.clientId));
    return {
      status: "ok",
      results: batch.map((b) => {
        rows.set(b.clientId, b);
        return { clientId: b.clientId, outcome: "saved" as const };
      }),
    };
  };
  return { rows, calls, push };
}

beforeEach(async () => {
  await closeOutbox();
  globalThis.indexedDB = new IDBFactory();
});
afterEach(closeOutbox);

describe("outbox", () => {
  it("a save is durable locally and touches no network until flushed", async () => {
    await enqueueBill("u1", bill("11111111-1111-4111-8111-111111111111"));
    expect(await outboxCounts("u1")).toEqual({ pending: 1, failed: 0 });
  });

  it("offline: a failed push keeps the entry, and the retry syncs it exactly once", async () => {
    const id = "11111111-1111-4111-8111-111111111111";
    await enqueueBill("u1", bill(id));

    const offline: PushFn = async () => {
      throw new TypeError("Failed to fetch");
    };
    expect(await flushOutbox("u1", offline)).toMatchObject({ stopped: "error", pushed: 0 });
    expect(await outboxCounts("u1")).toEqual({ pending: 1, failed: 0 });

    const server = fakeServer();
    expect(await flushOutbox("u1", server.push)).toMatchObject({ stopped: null, pushed: 1 });
    expect(server.rows.size).toBe(1);
    expect(await outboxCounts("u1")).toEqual({ pending: 0, failed: 0 });

    // Nothing left: a further flush makes no call at all.
    await flushOutbox("u1", server.push);
    expect(server.calls).toHaveLength(1);
  });

  it("re-saving the same session before sync replaces the entry (latest wins, one row)", async () => {
    const id = "11111111-1111-4111-8111-111111111111";
    await enqueueBill("u1", bill(id, "first"));
    await enqueueBill("u1", bill(id, "second"));
    const server = fakeServer();
    await flushOutbox("u1", server.push);
    expect(server.rows.size).toBe(1);
    expect(server.rows.get(id)?.billTitle).toBe("second");
  });

  it("an edit made while a push is in flight is NOT deleted by that push's success", async () => {
    const id = "11111111-1111-4111-8111-111111111111";
    await enqueueBill("u1", bill(id, "v1"));
    const server = fakeServer();
    let edited = false;
    const push: PushFn = async (batch) => {
      if (!edited) {
        edited = true;
        await enqueueBill("u1", bill(id, "v2")); // user edits mid-flight
      }
      return server.push(batch);
    };
    await flushOutbox("u1", push);
    // v1 landed, then the loop noticed v2 was still pending and pushed it too.
    expect(server.rows.get(id)?.billTitle).toBe("v2");
    expect(await outboxCounts("u1")).toEqual({ pending: 0, failed: 0 });
  });

  it("guest entries drain when someone logs in", async () => {
    await enqueueBill(null, bill("22222222-2222-4222-8222-222222222222"));
    const server = fakeServer();
    expect(await flushOutbox("u1", server.push)).toMatchObject({ pushed: 1 });
  });

  it("user A's unsynced bill is never pushed under user B", async () => {
    await enqueueBill("userA", bill("33333333-3333-4333-8333-333333333333"));
    const server = fakeServer();
    expect(await flushOutbox("userB", server.push)).toMatchObject({ pushed: 0 });
    expect(server.calls).toHaveLength(0);
    expect(await outboxCounts("userA")).toEqual({ pending: 1, failed: 0 });
    // ...and it is still there for A.
    expect(await flushOutbox("userA", server.push)).toMatchObject({ pushed: 1 });
  });

  it("an expired session stops the flush and keeps everything", async () => {
    await enqueueBill("u1", bill("11111111-1111-4111-8111-111111111111"));
    const res = await flushOutbox("u1", async () => ({ status: "unauthorized" }));
    expect(res.stopped).toBe("unauthorized");
    expect(await outboxCounts("u1")).toEqual({ pending: 1, failed: 0 });
  });

  it("a permanently rejected entry is parked as failed, not retried, and does not block others", async () => {
    await enqueueBill("u1", bill("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "bad"));
    await new Promise((r) => setTimeout(r, 2));
    await enqueueBill("u1", bill("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb", "good"));
    const push: PushFn = async (batch) => ({
      status: "ok",
      results: batch.map((b) =>
        b.billTitle === "bad"
          ? { clientId: b.clientId, outcome: "rejected" as const, error: "invalid" }
          : { clientId: b.clientId, outcome: "saved" as const },
      ),
    });
    expect(await flushOutbox("u1", push)).toMatchObject({ pushed: 1, rejected: 1, stopped: null });
    expect(await outboxCounts("u1")).toEqual({ pending: 0, failed: 1 });
    const server = fakeServer();
    await flushOutbox("u1", server.push);
    expect(server.calls).toHaveLength(0); // failed entry is not retried
  });

  it("concurrent flush calls share one run (no duplicate pushes)", async () => {
    await enqueueBill("u1", bill("11111111-1111-4111-8111-111111111111"));
    const server = fakeServer();
    await Promise.all([flushOutbox("u1", server.push), flushOutbox("u1", server.push)]);
    expect(server.calls).toHaveLength(1);
  });

  it("drains in batches", async () => {
    for (let i = 0; i < 5; i++) {
      await enqueueBill("u1", bill(`00000000-0000-4000-8000-00000000000${i}`));
    }
    const server = fakeServer();
    expect(await flushOutbox("u1", server.push, 2)).toMatchObject({ pushed: 5 });
    expect(server.calls.map((c) => c.length)).toEqual([2, 2, 1]);
  });
});
