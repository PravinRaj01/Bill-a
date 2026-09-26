import type { Cents, Receipt, SplitRecord } from "@/types/domain";

// The database and the split engine speak integer cents. The current UI
// (app/dashboard/new/page.tsx) still holds RM-as-float values from the old
// scan/split API. This file is the ONE boundary between them, so nothing else
// converts ad hoc. Phase 6 rewrites that page to be cents-native and this
// legacy half can be deleted — `toCents`/`fromCents`/`formatMoney` stay.

export const toCents = (amount: number): Cents =>
  Number.isFinite(amount) ? Math.round(amount * 100) : 0;

export const fromCents = (cents: Cents): number => cents / 100;

export const formatMoney = (cents: Cents, currency = "RM"): string =>
  `${currency}${fromCents(cents).toFixed(2)}`;

// --- legacy UI shapes (floats, snake_case) ---------------------------------

export interface LegacyReceiptItem {
  name: string;
  quantity: number;
  unit_price: number;
  total_price: number;
}
export interface LegacyReceiptData {
  items: LegacyReceiptItem[];
  tax: number;
  total: number;
  currency: string;
}
export interface LegacySplitRecord {
  name: string;
  amount: number;
  items: string;
}

export function receiptToDomain(ui: LegacyReceiptData): Receipt {
  return {
    items: ui.items.map((i) => ({
      name: i.name,
      quantity: i.quantity,
      unitPrice: toCents(i.unit_price),
      totalPrice: toCents(i.total_price),
    })),
    tax: toCents(ui.tax),
    total: toCents(ui.total),
    currency: ui.currency,
  };
}

export function receiptToLegacy(r: Receipt): LegacyReceiptData {
  return {
    items: r.items.map((i) => ({
      name: i.name,
      quantity: i.quantity,
      unit_price: fromCents(i.unitPrice),
      total_price: fromCents(i.totalPrice),
    })),
    tax: fromCents(r.tax),
    total: fromCents(r.total),
    currency: r.currency,
  };
}

export const splitsToDomain = (splits: LegacySplitRecord[]): SplitRecord[] =>
  splits.map((s) => ({ name: s.name, amount: toCents(s.amount), items: s.items }));

export const splitsToLegacy = (splits: SplitRecord[]): LegacySplitRecord[] =>
  splits.map((s) => ({ name: s.name, amount: fromCents(s.amount), items: s.items }));
