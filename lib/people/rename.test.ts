import { describe, expect, it } from "vitest";
import type { BillData } from "@/types/domain";
import { renamedForm, renameInBill, renameTransferKey } from "./rename";

const bill = (): BillData => ({
  split: [
    { name: "Alexis", amount: 4000, items: "a" },
    { name: "Pravin", amount: 2000, items: "b" },
  ],
  exactSplit: [
    { name: "Alexis", amount: 3998, items: "a" },
    { name: "pravin", amount: 2002, items: "b" },
  ],
  items: { items: [], tax: 0, total: 6000, currency: "RM" },
  people: ["Alexis", "Pravin"],
  reasoning: "r",
  cashRounding: { unit: 5, absorber: "Pravin" },
  settlement: { me: "Alexis", paidBy: [{ name: "Pravin", amount: 6000 }], paid: { "Alexis->Pravin": { at: "2026-01-01", amount: 4000 } }, dueDate: "2026-02-01" },
  participants: [{ name: "Alexis", me: true }, { name: "Pravin", userId: "u2" }],
});

describe("renamedForm", () => {
  it("maps the old full name to the new full name and the old first name to the new first name", () => {
    expect(renamedForm("Pravin Raj", "Pravin Raj", "Praveen Kumar")).toBe("Praveen Kumar");
    expect(renamedForm("pravin", "Pravin Raj", "Praveen Kumar")).toBe("Praveen");
  });
  it("leaves nicknames, unchanged names and cleared names alone", () => {
    expect(renamedForm("PJ", "Pravin Raj", "Praveen Kumar")).toBeNull();
    expect(renamedForm("Pravin", "Pravin Raj", "Pravin R")).toBeNull(); // first name didn't change
    expect(renamedForm("Pravin", "Pravin", "  pravin ")).toBeNull();
    expect(renamedForm("Pravin", "Pravin", null)).toBeNull();
    expect(renamedForm("Pravin", null, "Praveen")).toBeNull();
  });
});

describe("renameInBill", () => {
  it("renames the person everywhere the bill names them, and nothing else", () => {
    const out = renameInBill(bill(), "Pravin", "Praveen")!;
    expect(out.split.map((s) => s.name)).toEqual(["Alexis", "Praveen"]);
    expect(out.exactSplit!.map((s) => s.name)).toEqual(["Alexis", "Praveen"]);
    expect(out.people).toEqual(["Alexis", "Praveen"]);
    expect(out.cashRounding!.absorber).toBe("Praveen");
    expect(out.settlement!.me).toBe("Alexis");
    expect(out.settlement!.paidBy[0].name).toBe("Praveen");
    expect(out.settlement!.paid).toEqual({ "Alexis->Praveen": { at: "2026-01-01", amount: 4000 } });
    expect(out.settlement!.dueDate).toBe("2026-02-01");
    expect(out.participants).toEqual([{ name: "Alexis", me: true }, { name: "Praveen", userId: "u2" }]);
    expect(out.split.map((s) => s.amount)).toEqual([4000, 2000]);
  });
  it("does nothing when the person isn't on the bill, or the new name is already someone else on it", () => {
    expect(renameInBill(bill(), "Riley", "Sam")).toBeNull();
    expect(renameInBill(bill(), "Pravin", "alexis")).toBeNull();
    expect(renameInBill(bill(), "Pravin", "PRAVIN")).toBeNull();
  });
  it("keeps bills saved before settlement, participants or rounding existed", () => {
    const old: BillData = bill();
    delete old.settlement;
    delete old.participants;
    delete old.cashRounding;
    delete old.exactSplit;
    const out = renameInBill(old, "Pravin", "Praveen")!;
    expect(out.split[1].name).toBe("Praveen");
    expect("settlement" in out || "participants" in out || "cashRounding" in out).toBe(false);
  });
});

describe("renameTransferKey", () => {
  it("renames either side of a transfer", () => {
    expect(renameTransferKey("Pravin->Alexis", "pravin", "Praveen")).toBe("Praveen->Alexis");
    expect(renameTransferKey("Alexis->Pravin", "Pravin", "Praveen")).toBe("Alexis->Praveen");
    expect(renameTransferKey("Alexis->Riley", "Pravin", "Praveen")).toBe("Alexis->Riley");
  });
});
