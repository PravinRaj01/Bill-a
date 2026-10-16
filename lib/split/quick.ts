import type { Cents } from "@/types/domain";

// Quick split: an amount instead of a scanned receipt. The optional Service / SST chips add what a
// Malaysian restaurant bill adds, in the order it adds it: service charge on the amount, then SST
// on amount + service charge. Everything is integer cents, rounded once per line like a receipt.

export const SERVICE_RATE = 0.1;
export const SST_RATE = 0.06;

/** Largest amount accepted (RM99,999.99): well inside what the database and the engine take. */
export const MAX_QUICK_CENTS = 9_999_999;

export interface QuickTax {
  service: Cents;
  sst: Cents;
  /** service + sst: what goes in Receipt.tax. */
  tax: Cents;
  /** amount + tax. */
  total: Cents;
}

export function quickTax(amount: Cents, opts: { service?: boolean; sst?: boolean } = {}): QuickTax {
  const service = opts.service ? Math.round(amount * SERVICE_RATE) : 0;
  const sst = opts.sst ? Math.round((amount + service) * SST_RATE) : 0;
  return { service, sst, tax: service + sst, total: amount + service + sst };
}
