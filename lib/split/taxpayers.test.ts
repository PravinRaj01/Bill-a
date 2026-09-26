import { describe, expect, it } from "vitest";
import { validatePlan } from "@/lib/ai/validatePlan";
import { parseInstruction } from "./fallback-parser";

describe("validatePlan: taxPayers", () => {
  const ctx = { itemCount: 3, people: ["Pravin", "Aisha", "Sarah"] };
  const plan = (extra: Record<string, unknown> = {}) => ({ assignments: [], defaultRule: "equal", notes: "", ...extra });

  it("is passed through, canonicalised and deduped", () => {
    expect(validatePlan(plan({ taxPayers: ["aisha", "Aisha", "Sarah"] }), ctx)).toMatchObject({ ok: true, plan: { taxPayers: ["Aisha", "Sarah"] } });
  });

  it("an empty list means the default, and the key is omitted from the plan", () => {
    const r = validatePlan(plan({ taxPayers: [] }), ctx);
    expect(r.ok && "taxPayers" in r.plan).toBe(false);
  });

  it("a plan without the field (older answers) is still valid", () => {
    expect(validatePlan(plan(), ctx).ok).toBe(true);
  });

  it("drops unknown names but keeps the known one", () => {
    expect(validatePlan(plan({ taxPayers: ["Mallory", "Pravin"] }), ctx)).toMatchObject({ ok: true, plan: { taxPayers: ["Pravin"] } });
  });

  it("rejects a tax payer who is nobody in the bill (would silently change who pays)", () => {
    expect(validatePlan(plan({ taxPayers: ["Mallory"] }), ctx).ok).toBe(false);
  });
});

describe("parseInstruction: who pays the tax", () => {
  const two = ["Pravin", "Wifey"];
  const rf = [
    { name: "MUSH NOODLES DRY", quantity: 1, unitPrice: 1887, totalPrice: 1887 },
    { name: "STEAM DUMPLINGS", quantity: 1, unitPrice: 1604, totalPrice: 1604 },
  ];

  it.each([
    ["Wifey pays the tax", ["Wifey"]],
    ["Wifey pays for the service charge", ["Wifey"]],
    ["Wifey covers the GST", ["Wifey"]],
    ["put the SST on Wifey", ["Wifey"]],
    ["Pravin pays for the food and Wifey pays the tax", ["Wifey"]],
    ["Pravin and Wifey pay the tax", ["Pravin", "Wifey"]],
  ])("%s -> taxPayers %j", (text, expected) => {
    const r = parseInstruction(text, two, rf);
    expect(r.plan.taxPayers).toEqual(expected);
    expect(r.chips).toEqual([]); // and not "I couldn't work out what she had"
  });

  it("'Tax' is not mistaken for a person's name", () => {
    const r = parseInstruction("Pravin pays and Tax is on Wifey", two, rf);
    expect(r.chips.some((c) => c.kind === "unknown-person")).toBe(false);
  });

  it("names the items AND the tax payer in one clause", () => {
    const r = parseInstruction("Wifey pays for the noodles and the tax", two, rf);
    expect(r.plan.taxPayers).toEqual(["Wifey"]);
    expect(r.plan.assignments).toEqual([{ itemIndex: 0, people: ["Wifey"] }]);
  });

  it("'everything except the tax' does NOT make that person the tax payer", () => {
    const r = parseInstruction("Pravin pays for everything except the tax", two, rf);
    expect(r.plan.taxPayers).toBeUndefined();
    expect(r.plan.assignments).toHaveLength(2);
  });

  it("no tax mention -> no taxPayers key, exactly as before", () => {
    expect(parseInstruction("Pravin pays for the noodles", two, rf).plan).not.toHaveProperty("taxPayers");
  });
});
