import { describe, expect, it } from "vitest";

import { settlePill } from "./settle-pill";
import type { BillSettleSummary } from "@/lib/split/settle";

const base: BillSettleSummary = { owed: 6000, collected: 2000, outstanding: 4000, paidCount: 1, transferCount: 2, payers: ["Alexis"], mine: null, mismatch: false };
const today = "2026-10-17";

describe("settlePill", () => {
  it("is null for an untracked bill", () => {
    expect(settlePill({ settle: null }, today)).toBeNull();
  });
  it("shows what is outstanding and how many have paid", () => {
    expect(settlePill({ settle: base, currency: "RM" }, today)?.text).toBe("RM40.00 outstanding · 1 of 2 paid");
  });
  it("says settled, overdue, due today and due in N days", () => {
    expect(settlePill({ settle: { ...base, outstanding: 0 } }, today)?.text).toBe("Settled ✓");
    expect(settlePill({ settle: { ...base, dueDate: "2026-10-15" } }, today)?.text).toMatch(/overdue 2 days$/);
    expect(settlePill({ settle: { ...base, dueDate: today } }, today)?.text).toMatch(/due today$/);
    expect(settlePill({ settle: { ...base, dueDate: "2026-10-18" } }, today)?.text).toMatch(/due in 1 day$/);
  });
  it("flags a mismatch, and uses the bill's own currency", () => {
    expect(settlePill({ settle: { ...base, mismatch: true } }, today)?.text).toBe("Check who paid");
    expect(settlePill({ settle: base, currency: "Euro" }, today)?.text).toMatch(/^Euro 40\.00/);
  });
});
