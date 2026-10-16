import { describe, expect, it, vi } from "vitest";
import { planSplit } from "./planSplit";
import type { Receipt } from "@/types/domain";

// "Alexis 70%, Riley 30%" is arithmetic on typed numbers: it must be answered on the device,
// with no model call and no cross-check, even when a key is set.

const receipt: Receipt = {
  currency: "RM",
  tax: 1100,
  total: 11100,
  items: [
    { name: "FRIED RICE", quantity: 1, unitPrice: 4500, totalPrice: 4500 },
    { name: "CHICKEN RENDANG", quantity: 1, unitPrice: 3500, totalPrice: 3500 },
    { name: "TEH TARIK", quantity: 2, unitPrice: 1000, totalPrice: 2000 },
  ],
};
const people = ["Alexis", "Riley"];
const base = { receipt, people, applyTax: true, keys: { groq: "gsk_x" } as Record<string, string> };

describe("planSplit: whole-bill shares", () => {
  it("'sum up everything and alexis pays 70%, riley pay 30%' -> exact 70/30 of the bill, no network", async () => {
    const f = vi.fn();
    const out = await planSplit({ ...base, instructions: ["sum up everything and alexis pays 70%, riley pay 30%"], fetchImpl: f as never });
    expect(out.kind).toBe("shares");
    if (out.kind !== "shares") return;
    expect(out.result.splits.map((s) => [s.name, s.amount])).toEqual([["Alexis", 7770], ["Riley", 3330]]);
    expect(out.result.verified).toBe(true);
    expect(out.result.reasoning).toMatch(/Reconciled: 111\.00/);
    expect(f).not.toHaveBeenCalled();
  });

  it("uses the subtotal when tax & service is switched off", async () => {
    const f = vi.fn();
    const out = await planSplit({ ...base, applyTax: false, instructions: ["Alexis 50%, Riley 50%"], fetchImpl: f as never });
    expect(out.kind === "shares" && out.result.splits.reduce((s, r) => s + r.amount, 0)).toBe(10000);
    expect(f).not.toHaveBeenCalled();
  });

  it("includes a printed rounding adjustment in the bill", async () => {
    const f = vi.fn();
    const out = await planSplit({
      ...base,
      receipt: { ...receipt, total: 11098, rounding: -2 },
      instructions: ["Alexis 70%, Riley 30%"],
      fetchImpl: f as never,
    });
    expect(out.kind === "shares" && out.result.splits.reduce((s, r) => s + r.amount, 0)).toBe(11098);
  });

  it("a follow-up on the final page replaces the whole split", async () => {
    const f = vi.fn();
    const out = await planSplit({ ...base, instructions: ["split equally", "actually Alexis 70%, Riley 30%"], fetchImpl: f as never });
    expect(out.kind === "shares" && out.result.splits.map((s) => s.amount)).toEqual([7770, 3330]);
    expect(f).not.toHaveBeenCalled();
  });

  it("asks (no model call) when the percentages don't add up", async () => {
    const f = vi.fn();
    const out = await planSplit({ ...base, instructions: ["Alexis 70%, Riley 50%"], fetchImpl: f as never });
    expect(out.kind).toBe("needs-clarification");
    expect(out.kind === "needs-clarification" && out.chips[0]).toMatchObject({ kind: "shares" });
    expect(f).not.toHaveBeenCalled();
  });

  it("still catches a typo or a stranger BEFORE reading the shares", async () => {
    const f = vi.fn();
    const typo = await planSplit({ ...base, instructions: ["Alexs 70%, Riley 30%"], fetchImpl: f as never });
    expect(typo.kind === "needs-clarification" && typo.chips).toContainEqual({ kind: "possible-typo", token: "Alexs", suggestion: "Alexis" });
    const stranger = await planSplit({ ...base, instructions: ["Alexis 50%, Wei 50%"], fetchImpl: f as never });
    expect(stranger.kind === "needs-clarification" && stranger.chips).toContainEqual({ kind: "unknown-person", token: "Wei" });
    expect(f).not.toHaveBeenCalled();
  });

  it("an item instruction with a percentage is NOT treated as whole-bill shares", async () => {
    const f = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ choices: [{ message: { content: JSON.stringify({ assignments: [{ itemIndex: 0, people: ["Alexis"] }], defaultRule: "equal", notes: "" }) }, finish_reason: "stop" }] })),
    );
    const out = await planSplit({ ...base, instructions: ["Alexis pays 50% of the fried rice"], fetchImpl: f as never });
    expect(out.kind).not.toBe("shares");
    expect(f).toHaveBeenCalled();
  });
});
