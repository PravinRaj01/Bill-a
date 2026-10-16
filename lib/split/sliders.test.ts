import { describe, expect, it } from "vitest";
import {
  canMove,
  centsForPercent,
  equalShares,
  initialState,
  maxFor,
  percentOf,
  rebalance,
  rescale,
  setTyped,
  shareReducer,
  toggleLock,
  type ShareState,
} from "./sliders";

const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);
const st = (amounts: number[], locked: boolean[] = amounts.map(() => false)): ShareState => ({ amounts, locked });

describe("equalShares / initialState", () => {
  it("splits to the cent, odd cents to the first people", () => {
    expect(equalShares(1000, 3)).toEqual([334, 333, 333]);
    expect(equalShares(2335, 2)).toEqual([1168, 1167]);
    expect(equalShares(0, 3)).toEqual([0, 0, 0]);
    expect(equalShares(500, 0)).toEqual([]);
  });

  it("starts equal and unlocked; accepts a valid seed, rejects a bad one", () => {
    expect(initialState(1000, 2)).toEqual({ amounts: [500, 500], locked: [false, false] });
    expect(initialState(1000, 2, [700, 300]).amounts).toEqual([700, 300]);
    expect(initialState(1000, 2, [700, 200]).amounts).toEqual([500, 500]); // doesn't add up
    expect(initialState(1000, 3, [700, 300]).amounts).toEqual([334, 333, 333]); // wrong length
  });
});

describe("rebalance", () => {
  it("two people: the other takes the difference", () => {
    expect(rebalance(st([500, 500]), 0, 700).amounts).toEqual([700, 300]);
  });

  it("spreads the change over the others in proportion to what they hold", () => {
    // others hold 300 and 100 (3:1); person 0 drops 600 -> 200, 400 to give out, 300/100
    const out = rebalance(st([600, 300, 100]), 0, 200);
    expect(out.amounts).toEqual([200, 600, 200]);
  });

  it("clamps to 0 and to what locked people leave free", () => {
    expect(rebalance(st([500, 500]), 0, 9999).amounts).toEqual([1000, 0]);
    expect(rebalance(st([500, 500]), 0, -50).amounts).toEqual([0, 1000]);
    const locked = st([400, 300, 300], [false, true, false]);
    expect(maxFor(locked, 0)).toBe(700);
    expect(rebalance(locked, 0, 9999).amounts).toEqual([700, 300, 0]);
  });

  it("never moves a locked person, and a locked person can't be dragged", () => {
    const s = st([400, 300, 300], [false, true, false]);
    expect(rebalance(s, 0, 500).amounts[1]).toBe(300);
    expect(rebalance(st([400, 600], [true, false]), 0, 100)).toEqual(st([400, 600], [true, false]));
  });

  it("can't move when everyone else is locked", () => {
    const s = st([400, 600], [false, true]);
    expect(canMove(s, 0)).toBe(false);
    expect(rebalance(s, 0, 100)).toBe(s);
  });

  it("others holding nothing share the change equally", () => {
    expect(rebalance(st([1000, 0, 0]), 0, 400).amounts).toEqual([400, 300, 300]);
  });
});

describe("setTyped (a typed amount locks the person)", () => {
  it("sets the amount exactly and locks", () => {
    const out = setTyped(st([500, 500]), 0, 3050);
    expect(out.amounts).toEqual([1000, 0]); // clamped to the bill
    const out2 = setTyped(st([500, 500, 500]), 0, 300);
    expect(out2.amounts).toEqual([300, 600, 600]);
    expect(out2.locked).toEqual([true, false, false]);
  });

  it("two people: locks the typed one, the other is free", () => {
    const out = setTyped(st([500, 500]), 1, 300);
    expect(out).toEqual({ amounts: [700, 300], locked: [false, true] });
  });

  it("doesn't lock the last free person (nobody would be left to adjust)", () => {
    const out = setTyped(st([300, 300, 400], [true, false, false]), 1, 100);
    expect(out.amounts).toEqual([300, 100, 600]);
    expect(out.locked).toEqual([true, true, false]); // person 2 is still free
    const out2 = setTyped(st([300, 700], [true, false]), 1, 100);
    expect(out2).toEqual(st([300, 700], [true, false])); // nobody to absorb: no-op
  });
});

describe("toggleLock / rescale / percent helpers", () => {
  it("toggles", () => {
    expect(toggleLock(st([1, 1]), 1).locked).toEqual([false, true]);
  });

  it("rescale keeps proportions on a new total, exactly", () => {
    const out = rescale(st([700, 300], [true, false]), 2000);
    expect(out.amounts).toEqual([1400, 600]);
    expect(out.locked).toEqual([true, false]);
    expect(rescale(st([1, 1]), 2).amounts).toEqual([1, 1]);
    expect(sum(rescale(st([333, 333, 334]), 1235).amounts)).toBe(1235);
  });

  it("percentOf / centsForPercent", () => {
    expect(percentOf(7770, 11100)).toBe(70);
    expect(percentOf(1, 3)).toBe(33.3);
    expect(percentOf(5, 0)).toBe(0);
    expect(centsForPercent(70, 11100)).toBe(7770);
    expect(centsForPercent(150, 1000)).toBe(1000);
    expect(centsForPercent(NaN, 1000)).toBe(0);
  });
});

describe("shareReducer", () => {
  it("reset to equal also unlocks", () => {
    const out = shareReducer(st([700, 300], [true, false]), { type: "equal" });
    expect(out).toEqual({ amounts: [500, 500], locked: [false, false] });
  });

  it("restore puts back a snapshot (for Undo)", () => {
    const before = st([700, 300], [true, false]);
    const reset = shareReducer(before, { type: "equal" });
    expect(shareReducer(reset, { type: "restore", state: before })).toEqual(before);
  });
});

describe("property: any sequence of actions keeps the bill whole", () => {
  it("sum is the total, nothing negative, locked people never move on a drag", () => {
    let seed = 31337;
    const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
    for (let n = 0; n < 300; n++) {
      const people = 2 + Math.floor(rnd() * 6);
      const total = 1 + Math.floor(rnd() * 300000);
      let s = initialState(total, people);
      let currentTotal = total;
      for (let step = 0; step < 40; step++) {
        const i = Math.floor(rnd() * people);
        const r = rnd();
        if (r < 0.5) {
          const before = s;
          s = shareReducer(s, { type: "drag", i, cents: Math.floor(rnd() * (currentTotal + 200)) - 100 });
          before.amounts.forEach((a, j) => before.locked[j] && j !== i && expect(s.amounts[j]).toBe(a));
        } else if (r < 0.65) {
          s = shareReducer(s, { type: "type", i, cents: Math.floor(rnd() * currentTotal) });
        } else if (r < 0.8) {
          s = shareReducer(s, { type: "lock", i });
        } else if (r < 0.9) {
          currentTotal = 1 + Math.floor(rnd() * 300000);
          s = shareReducer(s, { type: "rescale", total: currentTotal });
        } else {
          s = shareReducer(s, { type: "equal" });
        }
        expect(sum(s.amounts)).toBe(currentTotal);
        s.amounts.forEach((a) => expect(a).toBeGreaterThanOrEqual(0));
        expect(s.amounts).toHaveLength(people);
      }
    }
  });
});
