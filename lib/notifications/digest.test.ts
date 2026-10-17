import { describe, expect, it } from "vitest";
import { dateInZone, digestSummary } from "./digest";
import type { PersonBalance } from "@/lib/split/settle";

const p = (name: string, currency: string, net: number, bills: PersonBalance["bills"] = []): PersonBalance => ({ key: `${currency}|${name}`, name, currency, net, bills });

describe("digestSummary", () => {
  it("is null when nobody owes anything", () => {
    expect(digestSummary([], "2026-10-17")).toBeNull();
  });

  it("names one person, counts several, and never nets currencies together", () => {
    expect(digestSummary([p("Riley", "RM", 4500)], "2026-10-17")).toBe("Riley owes you RM45.00");
    expect(digestSummary([p("Riley", "RM", 4500), p("Sam", "RM", 500), p("Lee", "SGD", -1000)], "2026-10-17")).toBe("2 people owe you RM50.00 · You owe Lee SGD 10.00");
  });

  it("says what is due today", () => {
    const bills = [{ id: "b", title: "t", createdAt: 0, amount: 4500, dueDate: "2026-10-17" }];
    expect(digestSummary([p("Riley", "RM", 4500, bills)], "2026-10-17")).toBe("Riley owes you RM45.00, due today");
  });
});

describe("dateInZone", () => {
  const t = new Date("2026-10-17T18:30:00Z");
  it("uses the person's own date, falling back to Malaysia for a bad zone", () => {
    expect(dateInZone(t, "Asia/Kuala_Lumpur")).toBe("2026-10-18");
    expect(dateInZone(t, "America/Los_Angeles")).toBe("2026-10-17");
    expect(dateInZone(t, "Not/AZone")).toBe("2026-10-18");
  });
});
