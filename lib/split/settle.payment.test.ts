import { describe, expect, it } from "vitest";

import { allOwedToMe, balanceNudgeText, groupNudgeText, markAllPaid, nudgeText, settleView, singlePayer, summaryText } from "./settle";
import type { Settlement } from "@/types/domain";

const PAY = "Pay me:\nDuitNow mobile number: 012-3456789";
const shares = [
  { name: "Alexis", amount: 3000 },
  { name: "Riley", amount: 3000 },
  { name: "Casey", amount: 3000 },
];
const settlement: Settlement = { me: "Alexis", paidBy: singlePayer("Alexis", 9000), paid: {} };
const view = (s: Settlement) => {
  const v = settleView(shares, s, "2026-10-16");
  if (!v || !v.ok) throw new Error("expected an ok view");
  return v;
};

describe("the payment block on shared messages", () => {
  it("is appended after the message when given, and absent otherwise", () => {
    const base = { debtor: "Riley", title: "Dinner", date: "2026-10-16", currency: "RM", owes: [{ to: "me", amount: 3000 }] };
    expect(nudgeText(base)).not.toContain("Pay me");
    expect(nudgeText({ ...base, payment: PAY }).endsWith(`via Bill.a\n\n${PAY}`)).toBe(true);
    // several creditors: still appended after the total
    expect(nudgeText({ ...base, owes: [{ to: "me", amount: 1000 }, { to: "Casey", amount: 2000 }], payment: PAY }).endsWith(`\n\n${PAY}`)).toBe(true);
  });
  it("applies to the group, balance and summary messages too", () => {
    const g = { title: "Dinner", date: "2026-10-16", currency: "RM", payTo: "me", lines: [{ debtor: "Riley", amount: 3000 }] };
    expect(groupNudgeText({ ...g, payment: PAY }).endsWith(`\n\n${PAY}`)).toBe(true);
    expect(groupNudgeText(g)).not.toContain("Pay me");
    const b = { name: "Riley", payTo: "me", currency: "RM", lines: [{ title: "Dinner", date: "2026-10-16", amount: 3000 }] };
    expect(balanceNudgeText({ ...b, payment: PAY }).endsWith(`\n\n${PAY}`)).toBe(true);
    expect(balanceNudgeText(b)).not.toContain("Pay me");
    const s = { currency: "RM", shares, settlement, view: view(settlement) };
    expect(summaryText({ ...s, payment: PAY }).endsWith(`Casey → Alexis: RM30.00\n\n${PAY}`)).toBe(true);
    expect(summaryText(s)).not.toContain("Pay me");
  });
});

describe("allOwedToMe", () => {
  it("is true only when something is still owed and all of it goes to me", () => {
    expect(allOwedToMe(view(settlement), "Alexis")).toBe(true);
    expect(allOwedToMe(view(settlement), " alexis ")).toBe(true);
    expect(allOwedToMe(view(settlement), "Riley")).toBe(false); // owed to Alexis, not Riley
    expect(allOwedToMe(view(settlement), undefined)).toBe(false);
    expect(allOwedToMe(null, "Alexis")).toBe(false);
  });
  it("is false when everything is settled, or when I owe someone too", () => {
    const done = markAllPaid(settlement, view(settlement).transfers, "2026-10-16");
    expect(allOwedToMe(view(done), "Alexis")).toBe(false);
    const split: Settlement = { me: "Alexis", paidBy: [{ name: "Alexis", amount: 5000 }, { name: "Riley", amount: 4000 }], paid: {} };
    expect(allOwedToMe(view(split), "Alexis")).toBe(false); // Casey owes Alexis and Riley
  });
});
