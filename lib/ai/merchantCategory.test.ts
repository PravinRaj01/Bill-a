import { describe, expect, it } from "vitest";
import { inferMerchantCategory } from "./merchantCategory";

const it_ = (name: string) => ({ name, quantity: 1, unitPrice: 100, totalPrice: 100 });

describe("inferMerchantCategory", () => {
  it.each([
    ["99 SPEEDMART", "Groceries"],
    ["MYDIN Subang Jaya", "Groceries"],
    ["Petronas Station", "Fuel"],
    ["Watsons Personal Care", "Health & Pharmacy"],
    ["TGV Cinemas", "Entertainment"],
    ["Grab", "Transport"],
    ["Restoran Nasi Kandar Pelita", "Food & Drink"],
    ["Starbucks", "Food & Drink"],
    ["Barber Shop", "Other"], // "bar" must not match inside "barber"
  ] as const)("merchant %s → %s", (name, expected) => {
    expect(inferMerchantCategory(name, [it_("ITEM")])).toBe(expected);
  });

  it("falls back to what was ordered when the merchant is unknown", () => {
    expect(inferMerchantCategory(null, [it_("TEH TARIK"), it_("NASI LEMAK"), it_("WIDGET")])).toBe("Food & Drink");
    expect(inferMerchantCategory("", [it_("USB CABLE"), it_("WIDGET")])).toBe("Other");
    expect(inferMerchantCategory(undefined, [])).toBe("Other");
  });

  it("a merchant keyword beats the item guess", () => {
    expect(inferMerchantCategory("99 Speedmart", [it_("TEH TARIK"), it_("NASI LEMAK")])).toBe("Groceries");
  });
});
