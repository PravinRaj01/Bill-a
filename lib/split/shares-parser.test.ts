import { describe, expect, it } from "vitest";
import { parseShareInstruction, type ShareParse } from "./shares-parser";
import { computeShareSplit } from "./shares";

const people = ["Pravin", "Lakx"];
const three = ["Pravin", "Lakx", "Aisha"];
const items = [
  { name: "FRIED RICE", quantity: 1, unitPrice: 900, totalPrice: 900 },
  { name: "TEH TARIK", quantity: 2, unitPrice: 400, totalPrice: 800 },
];
const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);

function shares(p: ShareParse) {
  if (p.kind !== "shares") throw new Error(`expected shares, got ${p.kind}: ${JSON.stringify(p)}`);
  return p;
}
function asks(p: ShareParse) {
  if (p.kind !== "ask") throw new Error(`expected ask, got ${p.kind}`);
  const chip = p.chips[0];
  if (chip.kind !== "shares") throw new Error("expected a shares chip");
  return chip;
}

describe("parseShareInstruction: percentages", () => {
  it("the user's own sentence: 'sum up everything and pravin pays 70%, lakx pay 30%'", () => {
    const r = shares(parseShareInstruction("sum up everything and pravin pays 70%, lakx pay 30%", people, 11100, items));
    expect(r.amounts).toEqual([7770, 3330]);
    expect(sum(r.amounts)).toBe(11100);
  });

  it.each([
    "Pravin 70%, Lakx 30%",
    "pravin pays 70% lakx pays 30%",
    "70% Pravin, 30% Lakx",
    "Pravin: 70 percent and Lakx: 30 percent",
    "Pravin pays 70 % / Lakx pays 30 %",
    "put 70% on Pravin and 30% on Lakx",
  ])("%s", (text) => {
    expect(shares(parseShareInstruction(text, people, 10000, items)).amounts).toEqual([7000, 3000]);
  });

  it("keeps decimals: 33.3% / 66.7% of RM100.00", () => {
    const r = shares(parseShareInstruction("Pravin 33.3%, Lakx 66.7%", people, 10000));
    expect(r.amounts).toEqual([3330, 6670]);
  });

  it("an indivisible total is still exact: 50/50 of RM0.01 + RM11.10", () => {
    const r = shares(parseShareInstruction("Pravin 50%, Lakx 50%", people, 1111));
    expect(sum(r.amounts)).toBe(1111);
    expect(r.amounts.slice().sort()).toEqual([555, 556]);
  });

  it("people not mentioned pay nothing when the percentages already make 100%", () => {
    const r = shares(parseShareInstruction("Pravin 70%, Lakx 30%", three, 10000));
    expect(r.amounts).toEqual([7000, 3000, 0]);
    expect(r.notes.join(" ")).toMatch(/Aisha isn't part of this split/);
  });

  it("'the rest equally': Pravin 50%, everyone else shares the other 50%", () => {
    const r = shares(parseShareInstruction("Pravin 50%, the rest equally", three, 10000));
    expect(r.amounts).toEqual([5000, 2500, 2500]);
  });

  it("'Lakx gets the rest': only the person named takes the remainder", () => {
    const r = shares(parseShareInstruction("Pravin 60%, Lakx gets the rest", three, 10000));
    expect(r.amounts).toEqual([6000, 4000, 0]);
  });
});

describe("parseShareInstruction: ratios", () => {
  it.each([
    ["70/30 Pravin and Lakx", [7000, 3000]],
    ["split 2:1 between Pravin and Lakx", [6667, 3333]],
    ["Lakx and Pravin 3:1", [2500, 7500]],
    ["split 50/50 between Pravin and Lakx", [5000, 5000]],
  ])("%s", (text, expected) => {
    expect(shares(parseShareInstruction(text, people, 10000)).amounts).toEqual(expected);
  });

  it("asks when the count of numbers and people differ", () => {
    expect(asks(parseShareInstruction("split 2:1:1 between Pravin and Lakx", people, 10000)).message).toMatch(/3 numbers.*2 people/);
  });
});

describe("parseShareInstruction: fixed amounts", () => {
  it("'Lakx pays RM30, the rest equally'", () => {
    const r = shares(parseShareInstruction("Lakx pays RM30, the rest equally", three, 11100));
    expect(r.amounts).toEqual([4050, 3000, 4050]);
  });

  it("RM amounts for everyone", () => {
    expect(shares(parseShareInstruction("Pravin RM60.50, Lakx RM50.50", people, 11100)).amounts).toEqual([6050, 5050]);
  });

  it("accepts '$' and 'ringgit'", () => {
    expect(shares(parseShareInstruction("Lakx pays $30, Pravin the rest", people, 11100)).amounts).toEqual([8100, 3000]);
    expect(shares(parseShareInstruction("Pravin 60 ringgit, Lakx the rest", people, 11100)).amounts).toEqual([6000, 5100]);
  });
});

describe("parseShareInstruction: asks instead of guessing", () => {
  it("percentages over 100%", () => {
    expect(asks(parseShareInstruction("Pravin 70%, Lakx 50%", people, 10000)).message).toMatch(/120%.*more than 100%/);
  });

  it("percentages under 100%, with a one-tap fix to share the rest", () => {
    const chip = asks(parseShareInstruction("Pravin 60%, Lakx 20%", three, 10000));
    expect(chip.message).toMatch(/80%.*20% is left over/);
    expect(chip.fixes).toEqual([{ label: "Share the other 20% equally (Aisha)", append: "the rest equally" }]);
  });

  it("applying the fix gives the answer", () => {
    const r = shares(parseShareInstruction("Pravin 60%, Lakx 20%, the rest equally", three, 10000));
    expect(r.amounts).toEqual([6000, 2000, 2000]);
  });

  it("someone named with no number: offers 'X gets the rest'", () => {
    const chip = asks(parseShareInstruction("Pravin 70%, Lakx", people, 10000));
    expect(chip.fixes[0].append).toBe("Lakx gets the rest");
  });

  it("a number with nobody to own it", () => {
    expect(asks(parseShareInstruction("Pravin 70% and 30%", people, 10000)).message).toMatch(/who the 30% is for/);
  });

  it("the same person twice", () => {
    expect(asks(parseShareInstruction("Pravin 70%, Pravin 30%", people, 10000)).message).toMatch(/two shares/);
  });

  it("amounts that exceed the bill, or don't cover it", () => {
    expect(asks(parseShareInstruction("Pravin RM90, Lakx RM30", people, 10000)).message).toMatch(/more than the bill/);
    const chip = asks(parseShareInstruction("Pravin RM30, Lakx RM20", three, 10000));
    expect(chip.message).toMatch(/RM50.00 is left over/);
    expect(chip.fixes[0].append).toBe("the rest equally");
  });

  it("mixing percentages and amounts", () => {
    expect(asks(parseShareInstruction("Pravin 70%, Lakx RM30", people, 10000)).message).toMatch(/not a mix/);
  });

  it("100% leaves nothing for 'the rest'", () => {
    expect(asks(parseShareInstruction("Pravin 100%, the rest equally", three, 10000)).message).toMatch(/nothing is left/);
  });
});

describe("parseShareInstruction: leaves item talk to the item pipeline", () => {
  it.each([
    "Pravin pays 50% of the fried rice",
    "split the teh tarik 70/30 between Pravin and Lakx",
    "Pravin pays the tax",
    "Lakx gets a 10% discount",
    "Pravin had the fried rice",
    "split equally",
    "Pravin and Lakx share the drinks",
    "I have 2 of them",
    "Pravin 70%", // handled as a share with one person: see below
  ])("%s", (text) => {
    const r = parseShareInstruction(text, people, 10000, items);
    if (text === "Pravin 70%") {
      expect(r.kind).toBe("ask"); // 70% only: 30% left over
    } else {
      expect(r.kind).toBe("none");
    }
  });

  it("numbers with nobody named are ignored", () => {
    expect(parseShareInstruction("70% 30%", people, 10000).kind).toBe("none");
  });
});

describe("computeShareSplit", () => {
  it("builds a verified SplitResult with a reasoning log", () => {
    const out = computeShareSplit(people, [7770, 3330], ["70% of the bill", "30% of the bill"], 11100, ["Shares you gave: Pravin 70%, Lakx 30%."]);
    expect(out.verified).toBe(true);
    expect(out.splits.map((s) => [s.name, s.amount])).toEqual([["Pravin", 7770], ["Lakx", 3330]]);
    expect(out.reasoning).toMatch(/Pravin: 77\.70 — 70% of the bill/);
    expect(out.reasoning).toMatch(/Reconciled: 111\.00 === bill 111\.00/);
  });

  it("refuses amounts that don't add up (a bug upstream, not user input)", () => {
    expect(() => computeShareSplit(people, [7000, 3000], ["", ""], 11100)).toThrow(/reconciliation/i);
  });
});

describe("property: whatever is parsed always reconciles to the bill", () => {
  it("random percentage splits across 2-6 people and random totals", () => {
    let seed = 2024;
    const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
    for (let n = 0; n < 500; n++) {
      const count = 2 + Math.floor(rnd() * 5);
      const names = Array.from({ length: count }, (_, i) => `Person${String.fromCharCode(65 + i)}`);
      const cuts = Array.from({ length: count - 1 }, () => Math.floor(rnd() * 99) + 1).sort((a, b) => a - b);
      const parts = [cuts[0], ...cuts.slice(1).map((c, i) => c - cuts[i]), 100 - cuts[cuts.length - 1]].filter((x) => x > 0);
      const used = names.slice(0, parts.length);
      const text = used.map((name, i) => `${name} ${parts[i]}%`).join(", ");
      const total = 1 + Math.floor(rnd() * 500000);
      const r = parseShareInstruction(text, names, total);
      expect(r.kind).toBe("shares");
      if (r.kind !== "shares") continue;
      expect(sum(r.amounts)).toBe(total);
      r.amounts.forEach((a, i) => {
        expect(a).toBeGreaterThanOrEqual(0);
        if (i >= used.length) expect(a).toBe(0);
        // within a cent of the exact share
        else expect(Math.abs(a - (total * parts[i]) / 100)).toBeLessThan(1);
      });
    }
  });
});
