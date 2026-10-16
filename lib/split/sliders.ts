import type { Cents } from "@/types/domain";
import { apportion } from "./apportion";

// The maths behind the per-person share sliders. State is each person's share in CENTS, always
// summing to the bill, so a typed amount never drifts and the footer can never say anything
// but "100%". Moving one person redistributes the difference over the UNLOCKED others in
// proportion to what they hold now (largest-remainder, so it stays exact). A typed amount also
// locks that person: it is the user's explicit number.

export interface ShareState {
  amounts: Cents[];
  locked: boolean[];
}

const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);

/** Everyone the same, to the cent (the odd cents go to the first people, like the engine). */
export function equalShares(total: Cents, count: number): Cents[] {
  return count > 0 ? apportion(total, Array(count).fill(1)) : [];
}

export function initialState(total: Cents, count: number, amounts?: Cents[]): ShareState {
  const ok = amounts && amounts.length === count && sum(amounts) === total && amounts.every((a) => a >= 0);
  return { amounts: ok ? [...amounts] : equalShares(total, count), locked: Array(count).fill(false) };
}

/** The most person `i` can hold: the bill minus what the locked others hold. */
export function maxFor(state: ShareState, i: number): Cents {
  const total = sum(state.amounts);
  return total - sum(state.amounts.filter((_, j) => j !== i && state.locked[j]));
}

/** True when someone else is free to absorb a change to person `i`. */
export function canMove(state: ShareState, i: number): boolean {
  return !state.locked[i] && state.amounts.some((_, j) => j !== i && !state.locked[j]);
}

/**
 * Person `i` asks for `wanted` cents. Clamped to [0, max]; the rest is spread over the free
 * others by their current shares (equally if they all hold 0). Locked people never move.
 */
export function rebalance(state: ShareState, i: number, wanted: Cents): ShareState {
  if (!canMove(state, i)) return state;
  const total = sum(state.amounts);
  const free = state.amounts.map((_, j) => j).filter((j) => j !== i && !state.locked[j]);
  const v = Math.min(Math.max(Math.round(wanted), 0), maxFor(state, i));
  const remaining = total - sum(state.amounts.filter((_, j) => j !== i && state.locked[j])) - v;
  const weights = free.map((j) => state.amounts[j]);
  const alloc = apportion(remaining, sum(weights) > 0 ? weights : weights.map(() => 1));
  const amounts = [...state.amounts];
  amounts[i] = v;
  free.forEach((j, k) => (amounts[j] = alloc[k]));
  return { amounts, locked: state.locked };
}

/** A typed amount: set it and lock the person (their number is now fixed). No-op if nobody could absorb it. */
export function setTyped(state: ShareState, i: number, wanted: Cents): ShareState {
  if (!canMove(state, i)) return state;
  const moved = rebalance(state, i, wanted);
  const locked = [...moved.locked];
  // Locking the last free person would leave nobody to adjust: only lock if someone else stays free.
  const othersFree = moved.amounts.some((_, j) => j !== i && !locked[j]);
  if (othersFree) locked[i] = true;
  return { amounts: moved.amounts, locked };
}

export function toggleLock(state: ShareState, i: number): ShareState {
  const locked = [...state.locked];
  locked[i] = !locked[i];
  return { amounts: state.amounts, locked };
}

/** The bill changed (an item or the tax was edited): keep everyone's proportions on the new total. */
export function rescale(state: ShareState, total: Cents): ShareState {
  if (sum(state.amounts) === total) return state;
  const count = state.amounts.length;
  const weights = sum(state.amounts) > 0 ? state.amounts : Array(count).fill(1);
  return { amounts: apportion(total, weights), locked: state.locked };
}

/** Whole-number percent when it is one, otherwise one decimal: 70, 33.3. */
export function percentOf(amount: Cents, total: Cents): number {
  if (total <= 0) return 0;
  return Math.round((amount / total) * 1000) / 10;
}

/** Cents for a typed percent (clamped to 0-100). */
export function centsForPercent(pct: number, total: Cents): Cents {
  if (!Number.isFinite(pct)) return 0;
  return Math.round((Math.min(Math.max(pct, 0), 100) / 100) * total);
}

export type ShareAction =
  | { type: "drag"; i: number; cents: Cents }
  | { type: "type"; i: number; cents: Cents }
  | { type: "lock"; i: number }
  | { type: "equal" }
  | { type: "rescale"; total: Cents }
  | { type: "restore"; state: ShareState };

export function shareReducer(state: ShareState, a: ShareAction): ShareState {
  switch (a.type) {
    case "drag":
      return rebalance(state, a.i, a.cents);
    case "type":
      return setTyped(state, a.i, a.cents);
    case "lock":
      return toggleLock(state, a.i);
    case "equal":
      return { amounts: equalShares(sum(state.amounts), state.amounts.length), locked: state.locked.map(() => false) };
    case "rescale":
      return rescale(state, a.total);
    case "restore":
      return a.state;
  }
}
