import { describe, expect, it, vi } from "vitest";
import { buildAssignmentPlanSchema } from "../schemas";
import { buildGroqBody, groqPlan } from "./groq";
import { buildGeminiBody, geminiPlan, geminiUrl, readGeminiText } from "./gemini";
import { explainAttempts, runCascade } from "./cascade";
import { testKey } from "./testKey";
import { ProviderError, kindForStatus, type PlanRequest } from "./types";
import { MODELS } from "./models";

const items = [
  { name: "NASI LEMAK", quantity: 1, unitPrice: 1000, totalPrice: 1000 },
  { name: "TEH TARIK", quantity: 2, unitPrice: 400, totalPrice: 800 },
  { name: "ROTI CANAI", quantity: 1, unitPrice: 300, totalPrice: 300 },
];
const req: PlanRequest = { people: ["Pravin", "Aisha"], items, instructions: ["Pravin pays for the teh tarik"] };

const goodPlan = { assignments: [{ itemIndex: 1, people: ["Pravin"] }], defaultRule: "equal", notes: "" };

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
const groqOk = (plan: unknown = goodPlan) => json({ choices: [{ message: { content: JSON.stringify(plan) }, finish_reason: "stop" }] });
const geminiOk = (plan: unknown = goodPlan) =>
  json({ candidates: [{ content: { parts: [{ text: JSON.stringify(plan) }] }, finishReason: "STOP" }] });

describe("schemas per provider", () => {
  const idx = [0, 1, 2];
  const people = ["Pravin", "Aisha"];

  it("groq strict: closed objects, every property required, integer enum", () => {
    const s = buildAssignmentPlanSchema(idx, people, "groq") as any;
    expect(s.additionalProperties).toBe(false);
    expect(s.required).toEqual(["assignments", "defaultRule", "taxPayers", "notes"]);
    expect(s.properties.taxPayers).toMatchObject({ type: "array", minItems: 0, maxItems: 2, items: { enum: people } });
    const item = s.properties.assignments.items;
    expect(item.additionalProperties).toBe(false);
    expect(item.required).toEqual(["itemIndex", "people"]);
    expect(item.properties.itemIndex).toEqual({ type: "integer", enum: idx });
    expect(item.properties.people).toMatchObject({ minItems: 1, maxItems: 2 });
    // Groq's live API rejects this keyword with HTTP 400 — must never come back.
    expect(JSON.stringify(s)).not.toContain("uniqueItems");
    // every key in `properties` is in `required` — the strict-mode rule
    for (const o of [s, item]) expect(Object.keys(o.properties).sort()).toEqual([...o.required].sort());
  });

  it("gemini: bounded integer instead of an integer enum, and no keywords its schema rejects", () => {
    const s = buildAssignmentPlanSchema(idx, people, "gemini") as any;
    expect(s).not.toHaveProperty("additionalProperties");
    const item = s.properties.assignments.items;
    expect(item).not.toHaveProperty("additionalProperties");
    expect(item.properties.itemIndex).toEqual({ type: "integer", minimum: 0, maximum: 2 });
    expect(item.properties.people).not.toHaveProperty("uniqueItems");
    expect(item.properties.people.items).toEqual({ type: "string", enum: people });
  });

  it("refuses to build a schema that could never be satisfied", () => {
    expect(() => buildAssignmentPlanSchema([], people)).toThrow();
    expect(() => buildAssignmentPlanSchema(idx, [])).toThrow();
  });
});

describe("request bodies", () => {
  it("groq: strict json_schema, low reasoning, deterministic, no prices in the prompt", () => {
    const body = buildGroqBody(req) as any;
    expect(body.model).toBe(MODELS.groq.primary);
    expect(body).toMatchObject({ temperature: 0, reasoning_effort: "low", include_reasoning: false });
    expect(body.response_format).toMatchObject({ type: "json_schema", json_schema: { name: "assignment_plan", strict: true } });
    const prompt = body.messages.map((m: any) => m.content).join("\n");
    expect(prompt).toContain("NASI LEMAK");
    expect(prompt).not.toMatch(/\b1000\b|\b10\.00\b|\b800\b/); // the LLM never sees a price
  });

  it("gemini: generationConfig carries the JSON mime type and schema; system prompt is separate", () => {
    const body = buildGeminiBody(req) as any;
    expect(body.generationConfig).toMatchObject({ temperature: 0, responseMimeType: "application/json" });
    expect(body.generationConfig.responseSchema.type).toBe("object");
    expect(body.systemInstruction.parts[0].text).toContain("never calculate money");
    expect(body.contents[0].role).toBe("user");
  });

  it("several instructions are applied in order", () => {
    const body = buildGroqBody({ ...req, instructions: ["split equally", "Aisha didn't have the rice"] }) as any;
    const user = body.messages[1].content as string;
    expect(user).toContain("apply in order");
    expect(user.indexOf("split equally")).toBeLessThan(user.indexOf("Aisha didn't"));
  });
});

describe("groqPlan", () => {
  it("sends the key only as a Bearer header to api.groq.com, never in the URL or body", async () => {
    const f = vi.fn().mockResolvedValue(groqOk());
    const out = await groqPlan(req, "gsk_SECRETKEY", { fetchImpl: f as never });
    expect(JSON.parse(out)).toEqual(goodPlan);
    const [url, init] = f.mock.calls[0];
    expect(url).toBe("https://api.groq.com/openai/v1/chat/completions");
    expect(url).not.toContain("SECRETKEY");
    expect((init as RequestInit).headers).toMatchObject({ Authorization: "Bearer gsk_SECRETKEY" });
    expect(String((init as RequestInit).body)).not.toContain("SECRETKEY");
  });

  it.each([
    [401, "auth"], [403, "auth"], [429, "rate-limit"], [404, "model-gone"], [400, "bad-request"], [500, "server"], [503, "server"],
  ])("HTTP %i -> %s", async (status, kind) => {
    const f = vi.fn().mockResolvedValue(json({ error: "x" }, status));
    await expect(groqPlan(req, "k", { fetchImpl: f as never })).rejects.toMatchObject({ kind, provider: "groq" });
  });

  it("a truncated (finish_reason length) or empty answer is bad-output", async () => {
    const cut = vi.fn().mockResolvedValue(json({ choices: [{ message: { content: "{" }, finish_reason: "length" }] }));
    await expect(groqPlan(req, "k", { fetchImpl: cut as never })).rejects.toMatchObject({ kind: "bad-output" });
    const empty = vi.fn().mockResolvedValue(json({ choices: [{ message: { content: null }, finish_reason: "stop" }] }));
    await expect(groqPlan(req, "k", { fetchImpl: empty as never })).rejects.toMatchObject({ kind: "bad-output" });
  });

  it("a network failure (offline / CORS / CSP block) becomes a network ProviderError", async () => {
    const f = vi.fn().mockRejectedValue(new TypeError("Failed to fetch"));
    await expect(groqPlan(req, "k", { fetchImpl: f as never })).rejects.toMatchObject({ kind: "network" });
  });
});

describe("geminiPlan", () => {
  it("sends the key only in x-goog-api-key, never in the URL", async () => {
    const f = vi.fn().mockResolvedValue(geminiOk());
    await geminiPlan(req, "AIzaSECRET", { fetchImpl: f as never });
    const [url, init] = f.mock.calls[0];
    expect(url).toBe(geminiUrl(MODELS.gemini.primary));
    expect(String(url)).not.toContain("AIzaSECRET");
    expect((init as RequestInit).headers).toMatchObject({ "x-goog-api-key": "AIzaSECRET" });
  });

  it("maps a safety block and an early stop to 'blocked', truncation to 'bad-output'", () => {
    expect(() => readGeminiText({ promptFeedback: { blockReason: "SAFETY" } })).toThrowError(/blocked/);
    expect(() => readGeminiText({ candidates: [{ finishReason: "SAFETY" }] })).toThrowError(ProviderError);
    try { readGeminiText({ candidates: [{ finishReason: "MAX_TOKENS", content: { parts: [{ text: "{" }] } }] }); }
    catch (e) { expect((e as ProviderError).kind).toBe("bad-output"); }
    expect(() => readGeminiText({ candidates: [{ finishReason: "STOP", content: { parts: [{ text: "ok" }] } }] })).not.toThrow();
  });

  it.each([[401, "auth"], [429, "rate-limit"], [404, "model-gone"], [500, "server"]])("HTTP %i -> %s", async (status, kind) => {
    const f = vi.fn().mockResolvedValue(json({}, status));
    await expect(geminiPlan(req, "k", { fetchImpl: f as never })).rejects.toMatchObject({ kind, provider: "gemini" });
  });
});

describe("Gemini's bad-key response (HTTP 400, not 401)", () => {
  const badKey = () =>
    new Response(JSON.stringify({ error: { code: 400, message: "API key not valid. Please pass a valid API key.", status: "INVALID_ARGUMENT" } }), { status: 400 });

  it("is reported as an auth problem, so the UI says 'rejected your key'", async () => {
    const f = vi.fn().mockResolvedValue(badKey());
    await expect(geminiPlan(req, "k", { fetchImpl: f as never })).rejects.toMatchObject({ kind: "auth", status: 400 });
  });

  it("but a 400 for any other reason stays a bad-request", async () => {
    const f = vi.fn().mockResolvedValue(new Response(JSON.stringify({ error: { message: "Invalid JSON payload" } }), { status: 400 }));
    await expect(geminiPlan(req, "k", { fetchImpl: f as never })).rejects.toMatchObject({ kind: "bad-request" });
  });

  it("the cascade explains it correctly", async () => {
    const f = vi.fn().mockResolvedValue(badKey());
    const r = await runCascade(req, { keys: { gemini: "AIza_x" }, fetchImpl: f as never });
    expect(explainAttempts(r.attempts)).toBe("Gemini rejected your key");
  });
});

describe("rate limits", () => {
  const limited = (headers: Record<string, string>) =>
    new Response("{}", { status: 429, headers });

  it("carries Retry-After (seconds) through as milliseconds", async () => {
    const f = vi.fn().mockResolvedValue(limited({ "retry-after": "12" }));
    await expect(groqPlan(req, "k", { fetchImpl: f as never })).rejects.toMatchObject({ kind: "rate-limit", retryAfterMs: 12_000 });
    await expect(geminiPlan(req, "k", { fetchImpl: f as never })).rejects.toMatchObject({ kind: "rate-limit", retryAfterMs: 12_000 });
  });

  it("ignores a missing, junk or absurd Retry-After (capped at 2 minutes)", async () => {
    for (const [h, expected] of [[{}, undefined], [{ "retry-after": "soon" }, undefined], [{ "retry-after": "99999" }, 120_000]] as const) {
      const f = vi.fn().mockResolvedValue(limited({ ...h }));
      await expect(groqPlan(req, "k", { fetchImpl: f as never })).rejects.toMatchObject({ retryAfterMs: expected });
    }
  });

  it("the cascade records it so a caller can wait and retry", async () => {
    const f = vi.fn().mockResolvedValue(limited({ "retry-after": "5" }));
    const r = await runCascade(req, { keys: { groq: "k" }, fetchImpl: f as never });
    expect(r.attempts[0]).toMatchObject({ kind: "rate-limit", retryAfterMs: 5_000 });
  });

  it("Groq reservation stays small: measured ~265 tokens worst case, and the free tier is 8,000/min", () => {
    expect((buildGroqBody(req) as any).max_completion_tokens).toBeLessThanOrEqual(1024);
  });
});

describe("Groq error detail", () => {
  it("keeps the provider's own explanation in the error, so a 400 is diagnosable", async () => {
    const f = vi.fn().mockResolvedValue(new Response(JSON.stringify({ error: { message: "uniqueItems is not supported" } }), { status: 400 }));
    await expect(groqPlan(req, "k", { fetchImpl: f as never })).rejects.toThrow(/uniqueItems is not supported/);
  });
});

describe("kindForStatus", () => {
  it("covers the statuses the cascade cares about", () => {
    expect(kindForStatus(408)).toBe("server");
    expect(kindForStatus(422)).toBe("bad-request");
  });
});

describe("runCascade", () => {
  const keys = { groq: "gsk_x", gemini: "AIza_x" };
  const routes = (groq: () => Promise<Response>, gemini: () => Promise<Response>) =>
    vi.fn((url: string) => (String(url).includes("groq.com") ? groq() : gemini()));

  it("uses Groq when it answers, and doesn't call Gemini at all", async () => {
    const f = routes(async () => groqOk(), async () => geminiOk());
    const r = await runCascade(req, { keys, fetchImpl: f as never });
    expect(r.tier).toBe("groq");
    expect(r.plan.assignments).toEqual([{ itemIndex: 1, people: ["Pravin"] }]);
    expect(f).toHaveBeenCalledTimes(1);
  });

  it.each([
    ["401 key rejected", 401, "auth"],
    ["429 rate limited", 429, "rate-limit"],
    ["500 server error", 500, "server"],
    ["404 model retired", 404, "model-gone"],
  ])("Groq %s -> falls through to Gemini", async (_n, status, kind) => {
    const f = routes(async () => json({}, status), async () => geminiOk());
    const r = await runCascade(req, { keys, fetchImpl: f as never });
    expect(r.tier).toBe("gemini");
    // Groq is tried with two models; both fail the same way here, then Gemini answers.
    expect(r.attempts.filter((a) => a.tier === "groq")).toHaveLength(kind === "auth" ? 1 : 2);
    expect(r.attempts[0]).toMatchObject({ tier: "groq", ok: false, kind });
    expect(r.attempts.at(-1)).toMatchObject({ tier: "gemini", ok: true });
  });

  it("an answer that fails validatePlan (unknown person) moves to the next tier", async () => {
    const bad = { assignments: [{ itemIndex: 1, people: ["Mallory"] }], defaultRule: "equal", notes: "" };
    const f = routes(async () => groqOk(bad), async () => geminiOk());
    const r = await runCascade(req, { keys, fetchImpl: f as never });
    expect(r.tier).toBe("gemini");
    expect(r.attempts[0]).toMatchObject({ tier: "groq", kind: "bad-output" });
  });

  it("an out-of-range item index is rejected", async () => {
    const bad = { assignments: [{ itemIndex: 9, people: ["Pravin"] }], defaultRule: "equal", notes: "" };
    const f = routes(async () => groqOk(bad), async () => geminiOk(bad));
    const r = await runCascade(req, { keys, fetchImpl: f as never });
    expect(r.tier).toBe("fallback");
  });

  it("a hung provider times out and the cascade moves on", async () => {
    const hang = (_u: string, init?: RequestInit) =>
      new Promise<Response>((_res, rej) => init?.signal?.addEventListener("abort", () => rej(Object.assign(new Error("aborted"), { name: "AbortError" }))));
    const f = vi.fn((url: string, init?: RequestInit) => (String(url).includes("groq.com") ? hang(url, init) : Promise.resolve(geminiOk())));
    const r = await runCascade(req, { keys, timeoutMs: 30, fetchImpl: f as never });
    expect(r.tier).toBe("gemini");
    expect(r.attempts[0]).toMatchObject({ tier: "groq", kind: "timeout" });
  });

  it("offline: both tiers fail on the network and the fallback parser answers", async () => {
    const f = vi.fn().mockRejectedValue(new TypeError("Failed to fetch"));
    const r = await runCascade(req, { keys, fetchImpl: f as never });
    expect(r.tier).toBe("fallback");
    expect(r.attempts.map((a) => a.kind)).toEqual(["network", "network", "network"]); // groq x2 models, gemini
    expect(r.plan.assignments).toEqual([{ itemIndex: 1, people: ["Pravin"] }]); // "Pravin pays for the teh tarik"
  });

  it("no keys at all: straight to the fallback, zero network calls", async () => {
    const f = vi.fn();
    const r = await runCascade(req, { keys: {}, fetchImpl: f as never });
    expect(r.tier).toBe("fallback");
    expect(r.attempts).toEqual([]);
    expect(f).not.toHaveBeenCalled();
  });

  it("a blank or whitespace key counts as no key", async () => {
    const f = vi.fn();
    await runCascade(req, { keys: { groq: "   ", gemini: "" }, fetchImpl: f as never });
    expect(f).not.toHaveBeenCalled();
  });

  it("only a Gemini key: Gemini is the first and only cloud tier", async () => {
    const f = routes(async () => groqOk(), async () => geminiOk());
    const r = await runCascade(req, { keys: { gemini: "AIza_x" }, fetchImpl: f as never });
    expect(r.tier).toBe("gemini");
    expect(f).toHaveBeenCalledTimes(1);
  });

  it("explains why the cloud tiers were skipped, in plain words", async () => {
    const f = routes(async () => json({}, 429), async () => json({}, 401));
    const r = await runCascade(req, { keys, fetchImpl: f as never });
    expect(explainAttempts(r.attempts)).toBe("Groq is rate-limited; Gemini rejected your key");
    expect(explainAttempts([])).toBeNull();
  });
});

describe("runCascade: Groq's two models and per-provider timeouts", () => {
  const keys = { groq: "gsk_x", gemini: "AIza_x" };

  it("tries the 20b first, then the 120b, using each model id", async () => {
    const models: string[] = [];
    const f = vi.fn(async (url: string, init?: RequestInit) => {
      if (String(url).includes("groq.com")) {
        models.push(JSON.parse(String(init?.body)).model);
        return models.length === 1 ? json({}, 429) : groqOk();
      }
      return geminiOk();
    });
    const r = await runCascade(req, { keys, fetchImpl: f as never });
    expect(models).toEqual([MODELS.groq.primary, MODELS.groq.secondary]);
    expect(MODELS.groq.primary).toBe("openai/gpt-oss-20b"); // the measured winner
    expect(r.tier).toBe("groq"); // the second Groq model answered; Gemini never needed
    expect(r.attempts.map((a) => [a.model, a.ok])).toEqual([[MODELS.groq.primary, false], [MODELS.groq.secondary, true]]);
  });

  it("a rejected key is not retried with the second Groq model", async () => {
    const f = vi.fn(async (url: string) => (String(url).includes("groq.com") ? json({}, 401) : geminiOk()));
    const r = await runCascade(req, { keys, fetchImpl: f as never });
    expect(f.mock.calls.filter(([u]) => String(u).includes("groq.com"))).toHaveLength(1);
    expect(r.tier).toBe("gemini");
  });

  it("can be limited to one Groq model", async () => {
    const f = vi.fn(async (url: string) => (String(url).includes("groq.com") ? json({}, 500) : geminiOk()));
    await runCascade(req, { keys, fetchImpl: f as never, groqSecondModel: false });
    expect(f.mock.calls.filter(([u]) => String(u).includes("groq.com"))).toHaveLength(1);
  });

  it("an explicit model (the bake-off) is used as-is, with no second model", async () => {
    const f = vi.fn(async () => json({}, 500));
    await runCascade(req, { keys: { groq: "gsk_x" }, models: { groq: "openai/gpt-oss-120b" }, fetchImpl: f as never });
    expect(f).toHaveBeenCalledTimes(1);
  });

  it("Gemini is given longer than Groq (it measured 3-12 s per call)", async () => {
    const { DEFAULT_TIMEOUTS } = await import("./cascade");
    expect(DEFAULT_TIMEOUTS.gemini).toBeGreaterThanOrEqual(15_000);
    expect(DEFAULT_TIMEOUTS.groq).toBeLessThan(DEFAULT_TIMEOUTS.gemini);
  });

  it("does not repeat itself when explaining two failures from the same provider", async () => {
    const f = vi.fn(async () => json({}, 429));
    const r = await runCascade(req, { keys: { groq: "gsk_x" }, fetchImpl: f as never });
    expect(explainAttempts(r.attempts)).toBe("Groq is rate-limited");
  });
});

describe("testKey", () => {
  it.each([
    [200, "valid"], [401, "invalid"], [403, "invalid"], [400, "invalid"], [429, "rate-limited"], [500, "unreachable"],
  ])("HTTP %i -> %s", async (status, expected) => {
    const f = vi.fn().mockResolvedValue(json({}, status));
    expect(await testKey("groq", "k", { fetchImpl: f as never })).toBe(expected);
  });

  it("lists models (no tokens spent) and keeps the key out of the URL", async () => {
    const f = vi.fn().mockResolvedValue(json({}));
    await testKey("gemini", "AIzaSECRET", { fetchImpl: f as never });
    const [url, init] = f.mock.calls[0];
    expect(String(url)).toContain("/models");
    expect(String(url)).not.toContain("AIzaSECRET");
    expect((init as RequestInit).headers).toMatchObject({ "x-goog-api-key": "AIzaSECRET" });
    expect((init as RequestInit).method).toBe("GET");
  });

  it("offline -> unreachable, not a crash", async () => {
    const f = vi.fn().mockRejectedValue(new TypeError("Failed to fetch"));
    expect(await testKey("groq", "k", { fetchImpl: f as never })).toBe("unreachable");
  });
});
