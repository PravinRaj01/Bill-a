import { describe, expect, it, vi } from "vitest";
import { planSplit } from "./planSplit";
import type { Receipt } from "@/types/domain";

const receipt: Receipt = {
  currency: "RM",
  tax: 0,
  total: 3000,
  items: [
    { name: "FRIED RICE", quantity: 1, unitPrice: 900, totalPrice: 900 },
    { name: "CHICKEN RENDANG", quantity: 1, unitPrice: 1300, totalPrice: 1300 },
    { name: "TEH TARIK", quantity: 2, unitPrice: 400, totalPrice: 800 },
  ],
};
const people = ["Pravin", "Aisha", "Sarah"];
const base = { receipt, people, applyTax: false, keys: {} as Record<string, string> };

const json = (b: unknown, status = 200) => new Response(JSON.stringify(b), { status });
const groqReturns = (plan: unknown) =>
  vi.fn().mockResolvedValue(json({ choices: [{ message: { content: JSON.stringify(plan) }, finish_reason: "stop" }] }));

describe("planSplit", () => {
  it("'add Wei' is routed to the people list — no LLM call, no split", async () => {
    const f = vi.fn();
    const out = await planSplit({ ...base, instructions: ["add Wei"], keys: { groq: "gsk_x" }, fetchImpl: f as never });
    expect(out).toEqual({ kind: "add-members", names: ["Wei"] });
    expect(f).not.toHaveBeenCalled();
  });

  it("asks about a typo BEFORE spending an LLM call", async () => {
    const f = vi.fn();
    const out = await planSplit({ ...base, instructions: ["Pravn had the fried rice"], keys: { groq: "gsk_x" }, fetchImpl: f as never });
    expect(out.kind).toBe("needs-clarification");
    expect(out.kind === "needs-clarification" && out.chips).toContainEqual({ kind: "possible-typo", token: "Pravn", suggestion: "Pravin" });
    expect(f).not.toHaveBeenCalled();
  });

  it("'continue anyway' skips the gate", async () => {
    const f = groqReturns({ assignments: [], defaultRule: "equal", notes: "" });
    const out = await planSplit({ ...base, instructions: ["Pravn had the fried rice"], keys: { groq: "gsk_x" }, ignoreAmbiguities: true, fetchImpl: f as never });
    expect(out.kind).toBe("split");
    expect(f).toHaveBeenCalledTimes(1);
  });

  it("with a working Groq key: the model's plan is computed by the ENGINE, to the cent", async () => {
    const f = groqReturns({ assignments: [{ itemIndex: 1, people: ["Pravin"] }], defaultRule: "equal", notes: "" });
    const out = await planSplit({ ...base, instructions: ["Pravin pays for the chicken"], keys: { groq: "gsk_x" }, fetchImpl: f as never });
    expect(out.kind).toBe("split");
    if (out.kind !== "split") return;
    expect(out.tier).toBe("groq");
    // 900 + 800 shared three ways (566.67 each -> 567/567/566), chicken 1300 all Pravin
    const by = Object.fromEntries(out.result.splits.map((s) => [s.name, s.amount]));
    expect(by.Pravin + by.Aisha + by.Sarah).toBe(3000);
    expect(by.Pravin).toBe(1300 + 567);
    expect(out.result.verified).toBe(true);
  });

  it("the model can't change the arithmetic: an answer with wrong-looking prices is irrelevant (prices aren't in the schema)", async () => {
    const f = groqReturns({ assignments: [{ itemIndex: 1, people: ["Pravin"] }], defaultRule: "equal", notes: "Pravin owes 9999.00" });
    const out = await planSplit({ ...base, instructions: ["Pravin pays for the chicken"], keys: { groq: "gsk_x" }, fetchImpl: f as never });
    expect(out.kind === "split" && out.result.splits.reduce((s, r) => s + r.amount, 0)).toBe(3000);
  });

  it("no key: the deterministic parser handles a clear instruction end to end", async () => {
    const out = await planSplit({ ...base, instructions: ["Pravin pays for the chicken, split the rest equally"] });
    expect(out.kind).toBe("split");
    if (out.kind !== "split") return;
    expect(out.tier).toBe("fallback");
    expect(out.result.splits.find((s) => s.name === "Pravin")!.amount).toBe(1300 + 567);
  });

  it("no key + something it can't resolve: a question with a preview, not a silent guess", async () => {
    const out = await planSplit({ ...base, instructions: ["Sarah had the tea"] });
    expect(out.kind).toBe("needs-clarification");
    if (out.kind !== "needs-clarification") return;
    expect(out.chips.length).toBeGreaterThan(0);
    expect(out.preview?.result.splits).toHaveLength(3);
  });

  it("a chat follow-up modifies the split: instructions accumulate, later wins", async () => {
    const out = await planSplit({
      ...base,
      instructions: ["Pravin pays for the chicken", "actually Aisha pays for the chicken"],
    });
    expect(out.kind).toBe("split");
    if (out.kind !== "split") return;
    expect(out.result.splits.find((s) => s.name === "Aisha")!.amount).toBeGreaterThanOrEqual(1300);
    expect(out.result.splits.find((s) => s.name === "Pravin")!.amount).toBeLessThan(1300);
  });

  it("an empty instruction means split equally", async () => {
    const out = await planSplit({ ...base, instructions: ["  "] });
    expect(out.kind).toBe("split");
    if (out.kind !== "split") return;
    const shares = out.result.splits.map((s) => s.amount);
    // each item is apportioned to the cent, so shares differ by a cent or two — but the total is exact
    expect(shares.reduce((a, b) => a + b, 0)).toBe(3000);
    for (const s of shares) expect(Math.abs(s - 1000)).toBeLessThanOrEqual(2);
  });

  it("rejects impossible input loudly", async () => {
    await expect(planSplit({ ...base, people: [], instructions: ["x"] })).rejects.toThrow();
    await expect(planSplit({ ...base, receipt: { ...receipt, items: [] }, instructions: ["x"] })).rejects.toThrow();
  });

  it("offline with keys: falls back, and says which cloud tiers failed", async () => {
    const f = vi.fn().mockRejectedValue(new TypeError("Failed to fetch"));
    const out = await planSplit({ ...base, instructions: ["Pravin pays for the chicken"], keys: { groq: "gsk_x", gemini: "AIza_x" }, fetchImpl: f as never });
    expect(out.kind).toBe("split");
    if (out.kind !== "split") return;
    expect(out.tier).toBe("fallback");
    expect(out.attempts.map((a) => a.kind)).toEqual(["network", "network", "network"]); // groq x2 models, gemini
  });
});
