import type { Cents } from "@/types/domain";

// The database, the split engine and (since the new-session page went cents-native) the UI
// all speak integer cents. Floats exist only at the edges: toCents() for values that arrive
// as RM floats (AI/OCR JSON, tests) and parseMoneyInput() for what a person types.

export const toCents = (amount: number): Cents =>
  Number.isFinite(amount) ? Math.round(amount * 100) : 0;

export const fromCents = (cents: Cents): number => cents / 100;

export const formatMoney = (cents: Cents, currency = "RM"): string =>
  `${currency}${fromCents(cents).toFixed(2)}`;

/** The text for a money <input>: "12.50" (no currency symbol). */
export const centsToInput = (cents: Cents): string => fromCents(cents).toFixed(2);

/**
 * What a person typed into a money field -> cents. Accepts "12.5" and "12,5"; anything that
 * isn't a non-negative number becomes 0 rather than NaN, so it can never poison a total.
 */
export function parseMoneyInput(raw: string): Cents {
  const n = parseFloat(raw.trim().replace(",", "."));
  return Number.isFinite(n) && n >= 0 ? Math.round(n * 100) : 0;
}
