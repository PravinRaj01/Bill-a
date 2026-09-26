import { afterEach, describe, expect, it, vi } from "vitest";
import { bucketConfidence, bucketItems, redact, telemetryEventSchema } from "./events";
import { POST } from "@/app/api/telemetry/route";

describe("redact", () => {
  it.each([
    ["Groq responded 401 for gsk_abcdefghijklmnop1234", "gsk_"],
    ["key AIzaSyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6 rejected", "AIza"],
    ["Authorization: Bearer abcdef123456", "abcdef123456"],
    ["x-goog-api-key: SECRETVALUE", "SECRETVALUE"],
    ["sk-proj-abcdefghijklmnopqrstuvwx", "abcdefghij"],
    ["token QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVowMTIzNDU2Nzg5", "QUJDREVG"],
  ])("removes credentials from %j", (text, secret) => {
    const out = redact(text);
    expect(out).not.toContain(secret);
    expect(out.length).toBeLessThanOrEqual(160);
  });

  it("leaves ordinary messages readable", () => {
    expect(redact("Groq responded 429: Rate limit reached")).toBe("Groq responded 429: Rate limit reached");
  });

  it("truncates long messages", () => {
    expect(redact("x ".repeat(300)).length).toBe(160);
  });
});

describe("buckets", () => {
  it("coarsens confidence to tenths and item counts to ranges", () => {
    expect(bucketConfidence(0.64)).toBe(6);
    expect(bucketConfidence(1.4)).toBe(10);
    expect(bucketConfidence(-1)).toBe(0);
    expect([0, 3, 5, 6, 15, 16, 40].map(bucketItems)).toEqual(["0", "1-5", "1-5", "6-15", "6-15", "16+", "16+"]);
  });
});

describe("event schema is a strict allowlist", () => {
  const split = { type: "split", tier: "groq", attempts: [], repairs: 0, ms: 512, rounds: 1 };

  it("accepts a well-formed event", () => {
    expect(telemetryEventSchema.safeParse(split).success).toBe(true);
    expect(telemetryEventSchema.safeParse({ type: "scan", engine: "tesseract", confidence: 7, items: "6-15", totalSource: "keyword", warnings: 1, ms: 800 }).success).toBe(true);
    expect(telemetryEventSchema.safeParse({ type: "error", where: "split", name: "Error", message: "boom", reconciliation: { expected: 100, actual: 101 } }).success).toBe(true);
  });

  it.each([
    ["an API key", { ...split, apiKey: "gsk_x" }],
    ["people's names", { ...split, people: ["Pravin"] }],
    ["item names", { ...split, items: ["NASI"] }],
    ["a price", { ...split, total: 1234 }],
    ["the instruction text", { ...split, instruction: "Pravin pays" }],
    ["a user id", { ...split, userId: "abc" }],
    ["an unknown event type", { type: "keylog", data: "x" }],
    ["a nested extra field", { ...split, attempts: [{ tier: "groq", kind: "auth", key: "gsk_x" }] }],
    ["a non-integer duration", { ...split, ms: 1.5 }],
    ["an unknown tier", { ...split, tier: "openai" }],
  ])("rejects %s", (_n, payload) => {
    expect(telemetryEventSchema.safeParse(payload).success).toBe(false);
  });

  it("rejects an over-long message", () => {
    expect(telemetryEventSchema.safeParse({ type: "error", where: "split", name: "E", message: "x".repeat(161) }).success).toBe(false);
  });
});

describe("POST /api/telemetry", () => {
  afterEach(() => vi.restoreAllMocks());
  const send = (body: string) => POST(new Request("http://x/api/telemetry", { method: "POST", body }));

  it("logs a valid event as one JSON line and answers 204", async () => {
    const log = vi.spyOn(console, "log").mockImplementation(() => {});
    const res = await send(JSON.stringify({ type: "split", tier: "gemini", attempts: [{ tier: "groq", kind: "rate-limit" }], repairs: 0, ms: 900, rounds: 2 }));
    expect(res.status).toBe(204);
    expect(log).toHaveBeenCalledTimes(1);
    expect(JSON.parse(String(log.mock.calls[0][0]))).toEqual({ telemetry: expect.objectContaining({ tier: "gemini", rounds: 2 }) });
  });

  it.each([
    ["not JSON", "{nope"],
    ["empty", ""],
    ["oversized", JSON.stringify({ type: "error", where: "split", name: "E", message: "x".repeat(3000) })],
    ["a smuggled key", JSON.stringify({ type: "split", tier: "groq", attempts: [], repairs: 0, ms: 1, rounds: 1, apiKey: "gsk_leak" })],
  ])("rejects %s with a bare 400 and logs nothing", async (_n, body) => {
    const log = vi.spyOn(console, "log").mockImplementation(() => {});
    const res = await send(body);
    expect(res.status).toBe(400);
    expect(await res.text()).toBe(""); // never echoes the payload back
    expect(log).not.toHaveBeenCalled();
  });
});
