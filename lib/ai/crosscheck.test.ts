import { describe, expect, it, vi } from "vitest";
import { planSplit } from "./planSplit";
import { parseInstruction } from "@/lib/split/fallback-parser";
import { telemetryEventSchema } from "@/lib/telemetry/events";
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
const base = { receipt, people, applyTax: false, keys: { groq: "gsk_x" } };

const json = (b: unknown) => new Response(JSON.stringify(b), { status: 200 });
const groqSays = (plan: unknown) =>
  vi.fn().mockResolvedValue(json({ choices: [{ message: { content: JSON.stringify(plan) }, finish_reason: "stop" }] }));
const plan = (assignments: unknown[], extra: Record<string, unknown> = {}) => ({ assignments, defaultRule: "equal", taxPayers: [], notes: "", ...extra });

describe("cross-check: cloud answer vs the on-device rules", () => {
  it("shows both readings when the AI and the rules disagree, each computed by the engine", async () => {
    // The rules read "Pravin pays for the chicken" correctly; the (mock) model gave it to Aisha.
    const f = groqSays(plan([{ itemIndex: 1, people: ["Aisha"] }]));
    const out = await planSplit({ ...base, instructions: ["Pravin pays for the chicken"], fetchImpl: f as never });
    expect(out.kind).toBe("disagreement");
    if (out.kind !== "disagreement") return;
    expect(out.ai.tier).toBe("groq");
    const amount = (r: { splits: { name: string; amount: number }[] }, n: string) => r.splits.find((s) => s.name === n)!.amount;
    expect(amount(out.ai.result, "Aisha")).toBeGreaterThanOrEqual(1300); // the AI's reading
    expect(amount(out.rules.result, "Pravin")).toBeGreaterThanOrEqual(1300); // the rules' reading
    // both are complete, engine-verified splits of the same bill
    for (const r of [out.ai.result, out.rules.result]) {
      expect(r.splits.reduce((s, x) => s + x.amount, 0)).toBe(3000);
      expect(r.verified).toBe(true);
    }
  });

  it("stays silent when they agree — the normal case costs the user nothing", async () => {
    const f = groqSays(plan([{ itemIndex: 1, people: ["Pravin"] }]));
    const out = await planSplit({ ...base, instructions: ["Pravin pays for the chicken"], fetchImpl: f as never });
    expect(out.kind).toBe("split");
  });

  it("agreement is judged on the AMOUNTS: a different but equivalent plan is not a disagreement", async () => {
    // AI spells out "everyone shares the rest" explicitly; the rules leave it as the default. Same money.
    const f = groqSays(plan([{ itemIndex: 1, people: ["Pravin"] }, { itemIndex: 0, people }, { itemIndex: 2, people }]));
    const out = await planSplit({ ...base, instructions: ["Pravin pays for the chicken, split the rest equally"], fetchImpl: f as never });
    expect(out.kind).toBe("split");
  });

  it("stays silent when the rules did not understand the instruction (nothing to judge against)", async () => {
    const instruction = "make it fair somehow";
    expect(parseInstruction(instruction, people, receipt.items).understood).toBe(false);
    // the AI does something specific; the blind default (equal) differs — but the rules had no opinion
    const f = groqSays(plan([{ itemIndex: 1, people: ["Aisha"] }]));
    const out = await planSplit({ ...base, instructions: [instruction], fetchImpl: f as never });
    expect(out.kind).toBe("split");
  });

  it("stays silent when the rules themselves are unsure (they have questions)", async () => {
    const f = groqSays(plan([{ itemIndex: 2, people: ["Sarah"] }]));
    const out = await planSplit({ ...base, instructions: ["Sarah had the tea"], ignoreAmbiguities: true, fetchImpl: f as never });
    expect(parseInstruction("Sarah had the tea", people, receipt.items).chips.length).toBeGreaterThan(0);
    expect(out.kind).toBe("split");
  });

  it("can be switched off (benchmarks that score the model alone)", async () => {
    const f = groqSays(plan([{ itemIndex: 1, people: ["Aisha"] }]));
    const out = await planSplit({ ...base, instructions: ["Pravin pays for the chicken"], crossCheck: false, fetchImpl: f as never });
    expect(out.kind).toBe("split");
  });

  it("never applies to the on-device tier itself (comparing the rules with themselves)", async () => {
    const out = await planSplit({ ...base, keys: {}, instructions: ["Pravin pays for the chicken"] });
    expect(out.kind).toBe("split");
    expect(out.kind === "split" && out.tier).toBe("fallback");
  });

  it("catches a tax-payer disagreement too (the AI ignored who pays the tax)", async () => {
    const taxed: Receipt = { ...receipt, tax: 300, total: 3300 };
    const f = groqSays(plan([]));
    const out = await planSplit({ ...base, receipt: taxed, applyTax: true, instructions: ["Aisha pays the tax"], fetchImpl: f as never });
    expect(out.kind).toBe("disagreement");
  });
});

describe("parseInstruction.understood", () => {
  const items = receipt.items;
  it.each([
    ["Split equally", true],
    ["Pravin pays for the chicken", true],
    ["Aisha pays the tax", true],
    ["Just split the rice between Pravin and Aisha, forget the rest", true],
    ["make it fair somehow", false],
    ["hello there", false],
    ["", false],
  ])("%j -> %s", (text, expected) => {
    expect(parseInstruction(text, people, items).understood).toBe(expected);
  });
});

describe("telemetry for the cross-check", () => {
  it("accepts shown / ai / rules, and nothing else", () => {
    for (const outcome of ["shown", "ai", "rules"]) {
      expect(telemetryEventSchema.safeParse({ type: "crosscheck", tier: "groq", outcome }).success).toBe(true);
    }
    expect(telemetryEventSchema.safeParse({ type: "crosscheck", tier: "groq", outcome: "dismissed" }).success).toBe(false);
    expect(telemetryEventSchema.safeParse({ type: "crosscheck", tier: "fallback", outcome: "ai" }).success).toBe(false);
  });

  it("carries no amounts, names or instruction text", () => {
    const withSplit = { type: "crosscheck", tier: "groq", outcome: "shown", amounts: [1, 2] };
    expect(telemetryEventSchema.safeParse(withSplit).success).toBe(false);
    expect(telemetryEventSchema.safeParse({ type: "crosscheck", tier: "groq", outcome: "ai", instruction: "x" }).success).toBe(false);
  });
});
