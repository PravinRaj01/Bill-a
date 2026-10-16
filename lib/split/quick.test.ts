import { describe, expect, it } from "vitest";
import { MAX_QUICK_CENTS, quickTax } from "./quick";

describe("quickTax", () => {
  it("is just the amount with no chips", () => {
    expect(quickTax(10000)).toEqual({ service: 0, sst: 0, tax: 0, total: 10000 });
  });

  it("service 10% on the amount", () => {
    expect(quickTax(10000, { service: true })).toEqual({ service: 1000, sst: 0, tax: 1000, total: 11000 });
  });

  it("SST 6% on the amount alone", () => {
    expect(quickTax(10000, { sst: true })).toEqual({ service: 0, sst: 600, tax: 600, total: 10600 });
  });

  it("both: SST is charged on amount + service charge, as a restaurant bill does", () => {
    // 100.00 + 10.00 service = 110.00; SST 6% of 110.00 = 6.60
    expect(quickTax(10000, { service: true, sst: true })).toEqual({ service: 1000, sst: 660, tax: 1660, total: 11660 });
  });

  it("always whole cents and total = amount + tax", () => {
    let seed = 5;
    const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
    for (let n = 0; n < 500; n++) {
      const amount = 1 + Math.floor(rnd() * MAX_QUICK_CENTS);
      const q = quickTax(amount, { service: rnd() < 0.5, sst: rnd() < 0.5 });
      expect(Number.isInteger(q.service) && Number.isInteger(q.sst) && Number.isInteger(q.total)).toBe(true);
      expect(q.total).toBe(amount + q.tax);
      expect(q.tax).toBe(q.service + q.sst);
    }
  });

  it("the maximum stays inside what the database accepts (100,000,000 cents)", () => {
    expect(quickTax(MAX_QUICK_CENTS, { service: true, sst: true }).total).toBeLessThan(100_000_000);
  });
});
