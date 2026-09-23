// Shared domain types for Bill.a. These replace the inline, non-exported
// interfaces that used to live in app/dashboard/new/page.tsx (ReceiptItem,
// ReceiptData, SplitRecord) and the "any"-typed Supabase rows scattered
// across the dashboard pages.
//
// The one rule that matters everywhere in this file: money is always an
// integer number of cents (or sen, or whatever the currency's minor unit
// is). Never a float. See lib/split/engine.ts for why.

/** An integer count of the currency's minor unit (cents/sen/etc). Never a float. */
export type Cents = number;

export interface ReceiptItem {
  name: string;
  quantity: number;
  unitPrice: Cents;
  totalPrice: Cents;
}

export interface Receipt {
  items: ReceiptItem[];
  /** Combined tax + service charge as printed on the receipt, if separately itemized. */
  tax: Cents;
  /** Grand total as printed on the receipt. This is the number every split must reconcile to. */
  total: Cents;
  currency: string;
}

/**
 * The ONLY thing an LLM (or the fallback parser) is ever allowed to
 * produce. It assigns receipt items to people; it never computes a price.
 * See lib/split/engine.ts for where the actual arithmetic happens, and
 * lib/ai/schemas.ts for how this shape is turned into a grammar that
 * makes most of these constraints structurally unavoidable.
 */
export interface AssignmentPlan {
  assignments: Assignment[];
  /** What happens to an item nobody explicitly mentioned. */
  defaultRule: "equal" | "exclude";
  /** Free-text explanation from the model, surfaced for debugging, never trusted for math. */
  notes: string;
}

export interface Assignment {
  itemIndex: number;
  people: string[];
  /**
   * Optional, parallel to `people`. Omit for an even split among the
   * listed people. When present, must be the same length as `people`.
   */
  weights?: number[];
}

export interface SplitRecord {
  name: string;
  amount: Cents;
  /** Human-readable summary of what this person is paying for, e.g. "Iced Tea (x1), Nasi Lemak (x0.5)". */
  items: string;
}

export interface SplitResult {
  splits: SplitRecord[];
  /** Generated FROM the actual computation (lib/split/engine.ts), never model prose. */
  reasoning: string;
  /** Always true if computeSplit() returned without throwing — the reconciliation invariant held. */
  verified: true;
}

export class SplitReconciliationError extends Error {
  constructor(
    public readonly expected: Cents,
    public readonly actual: Cents,
  ) {
    super(
      `Split reconciliation failed: expected total ${expected} cents, got ${actual} cents ` +
        `(diff ${actual - expected}). This should be unreachable given apportion()'s invariant — ` +
        `if you see this, apportion() or computeSplit() has a bug, not the input data.`,
    );
    this.name = "SplitReconciliationError";
  }
}
