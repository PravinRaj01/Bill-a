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
  /**
   * Signed cash-rounding adjustment printed on the receipt ("Rounding Adj -0.02"). Already
   * included in `total`. Absent on receipts without one and on bills saved before it existed.
   */
  rounding?: Cents;
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
  /**
   * Who pays the tax / service charge, shared equally among them. Omitted or empty means the
   * default: everyone pays it in proportion to what they ordered. Only the ENGINE turns this
   * into money; the model just names the people.
   */
  taxPayers?: string[];
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

/**
 * The single shape stored in bill_history.data. Replaces the two legacy
 * shapes the old pages defensively unwrapped (a bare array, or
 * `{ splits }` vs `{ split }`) — with a clean-slate database there is
 * nothing to stay compatible with.
 */
export interface BillData {
  split: SplitRecord[];
  items: Receipt;
  people: string[];
  reasoning: string;
  /** Cash rounding applied to `split`: `unit` in cents (5 = nearest 5 sen), `absorber` takes the difference. Absent = exact. */
  cashRounding?: { unit: number; absorber: string };
  /** The exact shares before cash rounding, kept so the rounding can be changed or undone after a restore. */
  exactSplit?: SplitRecord[];
  /** Who paid and who has paid back (batch 2). */
  settlement?: Settlement;
}

/**
 * Who fronted a bill and who has paid them back. Optional on a saved bill (older bills have none).
 * Transfers ("Lakx -> Pravin RM41.20") are DERIVED from the shares and `paidBy`, never stored: only
 * the facts people entered are. See lib/split/settle.ts.
 */
export interface Settlement {
  /** Which person is the app's user on this bill (used for "you're owed" and balances). */
  me?: string;
  /** Who paid the bill and how much each fronted. One entry = a single payer (amount = the bill). */
  paidBy: { name: string; amount: Cents }[];
  /**
   * Transfers marked as paid, keyed "From->To". The amount is stored with the mark so it stops
   * counting if the shares (and so the amount owed) change later.
   */
  paid: Record<string, { at: string; amount: Cents }>;
  /** "YYYY-MM-DD", the local date by which people should have paid. */
  dueDate?: string;
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
