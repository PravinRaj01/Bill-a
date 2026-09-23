import { describe, expect, it } from "vitest";
import {
  detectDefaultRule,
  extractExceptionClause,
  resolveItemReference,
  resolveInstruction,
  complement,
  isInformativeDefaultRule,
  FALLBACK_DEFAULT_REASON,
} from "./resolve";
import { classifyItem, matchCategoryPhrase } from "./lexicon";
import type { ReceiptItem } from "@/types/domain";

const item = (name: string): ReceiptItem => ({
  name,
  quantity: 1,
  unitPrice: 100,
  totalPrice: 100,
});

// The item list from the bake-off's drinks-payer case — the single case
// that failed on every model tested.
const DRINKS_CASE = [
  item("NASI LEMAK SPECIAL"),
  item("ICED LEMON TEA"),
  item("TEH TARIK"),
  item("CHICKEN RENDANG"),
];

describe("lexicon — item classification", () => {
  it.each([
    ["TEH TARIK", "drinks"],
    ["ICED LEMON TEA", "drinks"],
    ["KOPI O", "drinks"],
    ["MILO AIS", "drinks"],
    ["SIRAP BANDUNG", "drinks"],
    ["MILKSHAKE", "drinks"],
    ["CHOCOLATE LAVA CAKE", "desserts"],
    ["LEMON SORBET", "desserts"],
    ["NASI LEMAK SPECIAL", "mains"],
    ["CHICKEN RENDANG", "mains"],
    ["MEE GORENG", "mains"],
    ["FRENCH FRIES", "sides"],
    ["GARLIC BREAD", "sides"],
    ["TIGER BEER", "alcohol"],
  ])("classifies %s as %s", (name, expected) => {
    const scores = classifyItem(name);
    expect(scores[0]?.categoryId).toBe(expected);
  });

  it("does not let a sweet-sounding token drag a drink into desserts", () => {
    // "MILKSHAKE" contains no dessert keyword, but this guards the class
    // of bug where a shared token (chocolate, lemon) wins over the real
    // category signal.
    expect(classifyItem("CHOCOLATE MILKSHAKE")[0].categoryId).toBe("drinks");
  });

  it("prefers a multi-word phrase match over a single-token match", () => {
    // "ice cream" (phrase, desserts) should beat "cream" appearing alone.
    expect(classifyItem("VANILLA ICE CREAM")[0].categoryId).toBe("desserts");
  });
});

describe("lexicon — user phrase matching", () => {
  it.each([
    ["the drinks", "drinks"],
    ["drinks", "drinks"],
    ["beverages", "drinks"],
    ["the desserts", "desserts"],
    ["anything sweet", "desserts"],
    ["the beers", "alcohol"],
    ["the sides", "sides"],
  ])("maps %s to %s", (phrase, expected) => {
    expect(matchCategoryPhrase(phrase)).toBe(expected);
  });

  it("returns null for a phrase that names no category", () => {
    expect(matchCategoryPhrase("pravin")).toBeNull();
  });
});

describe("detectDefaultRule", () => {
  it.each([
    ["Pravin pays for the drinks, split the rest equally", "equal"],
    ["Split everything equally", "equal"],
    ["Everyone splits equally except the milkshake, which is just for Sarah", "equal"],
    ["Pravin had the teh tarik, split the rest between everyone", "equal"],
    ["Just split the burger between Pravin and Sarah, forget about the rest of the table's order", "exclude"],
    ["Only split the pizza, ignore the rest", "exclude"],
    ["Split the starters, nothing else", "exclude"],
  ])("%s -> %s", (instruction, expected) => {
    expect(detectDefaultRule(instruction).defaultRule).toBe(expected);
  });

  it('does not treat "just for <person>" as an exclude marker', () => {
    // This is the trap: "just" here means an item belongs exclusively to
    // someone, NOT that the rest of the bill is excluded. Getting this
    // wrong would silently leave part of the bill unpaid.
    const r = detectDefaultRule("Everyone splits equally except the milkshake, which is just for Sarah");
    expect(r.defaultRule).toBe("equal");
  });

  it("defaults to equal when no marker is present, and says so", () => {
    const r = detectDefaultRule("Pravin and Aisha had the pizza");
    expect(r.defaultRule).toBe("equal");
    expect(r.reason).toContain("defaulting");
  });
});

describe("extractExceptionClause", () => {
  it.each([
    ["Everyone splits equally except the milkshake", "the milkshake"],
    ["Split it all apart from the beers", "the beers"],
    ["Everything other than the dessert", "the dessert"],
    ["All but the coffee", "the coffee"],
  ])("%s -> %s", (instruction, expected) => {
    expect(extractExceptionClause(instruction)).toContain(expected);
  });

  it("returns null when there is no exception", () => {
    expect(extractExceptionClause("Split everything equally")).toBeNull();
  });
});

describe("resolveItemReference", () => {
  it("resolves a category phrase to every matching item — the case every model failed", () => {
    const ref = resolveItemReference("the drinks", DRINKS_CASE);
    expect(ref).not.toBeNull();
    // ICED LEMON TEA (1) and TEH TARIK (2), not the rice or the rendang.
    expect(ref!.itemIndices.sort()).toEqual([1, 2]);
    expect(ref!.via).toBe("category");
  });

  it("resolves an exact item name", () => {
    const ref = resolveItemReference("teh tarik", DRINKS_CASE);
    expect(ref!.itemIndices).toEqual([2]);
    expect(ref!.confidence).toBe(1);
  });

  it("flags genuine ambiguity with low confidence instead of guessing", () => {
    // "the tea" matches both ICED LEMON TEA and TEH TARIK... except only
    // one contains the token "tea", so this should resolve cleanly.
    const ref = resolveItemReference("the lemon tea", DRINKS_CASE);
    expect(ref!.itemIndices).toEqual([1]);
  });

  it("returns low confidence when several items tie on token overlap", () => {
    const items = [item("CHICKEN RICE"), item("CHICKEN CURRY")];
    const ref = resolveItemReference("the chicken", items);
    expect(ref!.itemIndices.sort()).toEqual([0, 1]);
    expect(ref!.confidence).toBeLessThan(0.5);
  });

  it("returns null when nothing matches at all", () => {
    expect(resolveItemReference("the helicopter", DRINKS_CASE)).toBeNull();
  });
});

describe("complement", () => {
  it("returns everything not listed", () => {
    expect(complement([1, 2], 4)).toEqual([0, 3]);
  });
  it("returns everything when nothing is claimed", () => {
    expect(complement([], 3)).toEqual([0, 1, 2]);
  });
  it("returns nothing when everything is claimed", () => {
    expect(complement([0, 1], 2)).toEqual([]);
  });
});

describe("resolveInstruction — the five cases that failed on every model", () => {
  it("drinks-payer: resolves the drinks AND computes the rest", () => {
    const r = resolveInstruction(
      "Pravin pays for the drinks, split the rest equally",
      DRINKS_CASE,
    );
    const drinks = r.references.find((x) => x.via === "category");
    expect(drinks!.itemIndices.sort()).toEqual([1, 2]);

    const rest = r.references.find((x) => x.via === "complement");
    expect(rest!.itemIndices.sort()).toEqual([0, 3]);

    expect(r.defaultRule).toBe("equal");
    expect(r.promptBlock).toContain("items [1, 2]");
  });

  it("category-desserts: resolves desserts against a mixed list", () => {
    const items = [
      item("BURGER DELUXE"),
      item("FRENCH FRIES"),
      item("CHOCOLATE LAVA CAKE"),
      item("LEMON SORBET"),
    ];
    const r = resolveInstruction("Sarah pays for all the desserts, split the rest equally", items);
    const desserts = r.references.find((x) => x.via === "category");
    expect(desserts!.itemIndices.sort()).toEqual([2, 3]);
    expect(r.defaultRule).toBe("equal");
  });

  it("exclude-unmentioned-items: detects the exclude rule that every model got wrong", () => {
    const items = [item("BURGER DELUXE"), item("FISH AND CHIPS"), item("MILKSHAKE")];
    const r = resolveInstruction(
      "Just split the burger between Pravin and Sarah, forget about the rest of the table's order",
      items,
    );
    expect(r.defaultRule).toBe("exclude");
  });

  it("informal-item-reference: resolves a lowercase mention to the right item", () => {
    const r = resolveInstruction(
      "Pravin had the teh tarik, split the rest between everyone",
      DRINKS_CASE,
    );
    const tt = r.references.find((x) => x.itemIndices.length === 1 && x.itemIndices[0] === 2);
    expect(tt).toBeDefined();
    const rest = r.references.find((x) => x.via === "complement");
    expect(rest!.itemIndices).not.toContain(2);
    expect(r.defaultRule).toBe("equal");
  });

  it("negation-one-item: marks the excepted item and keeps defaultRule equal", () => {
    const items = [item("FRIES"), item("NUGGETS"), item("MILKSHAKE")];
    const r = resolveInstruction(
      "Everyone splits equally except the milkshake, which is just for Sarah",
      items,
    );
    const shake = r.references.find((x) => x.itemIndices.includes(2));
    expect(shake).toBeDefined();
    expect(shake!.isException).toBe(true);
    // The critical bit: "just for Sarah" must NOT flip this to exclude.
    expect(r.defaultRule).toBe("equal");
  });
});

describe("resolveInstruction — fails open, and stays silent when it has nothing informative to say", () => {
  it("resolves no items and emits nothing when defaultRule is only the uninformative fallback", () => {
    const r = resolveInstruction("do the thing with the stuff", DRINKS_CASE);
    expect(r.references).toEqual([]);
    expect(r.defaultRule).toBe("equal");
    expect(r.defaultRuleReason).toBe(FALLBACK_DEFAULT_REASON);
    // Regression found in the bake-off: emitting "DEFAULT RULE: equal"
    // here — true, but zero-information since it's already the model's
    // own stated fallback — measurably hurt Llama-3.2-1B on this exact
    // shape of case. See isInformativeDefaultRule's doc comment.
    expect(r.promptBlock).toBe("");
  });

  it("omits low-confidence references, and stays silent when the default rule isn't informative either", () => {
    const items = [item("CHICKEN RICE"), item("CHICKEN CURRY")];
    const r = resolveInstruction("Pravin had the chicken", items);
    // Ambiguous between the two -> must not be asserted to the model as fact.
    expect(r.promptBlock).not.toContain("items [");
    expect(r.promptBlock).toBe("");
  });

  it("still surfaces DEFAULT RULE: exclude — the one case proven to need it", () => {
    const items = [item("BURGER DELUXE"), item("FISH AND CHIPS"), item("MILKSHAKE")];
    const r = resolveInstruction(
      "Just split the burger between Pravin and Sarah, forget about the rest of the table's order",
      items,
    );
    expect(r.promptBlock).toContain("DEFAULT RULE: exclude");
  });

  it("still surfaces an EXPLICIT equal marker, distinct from the uninformative fallback", () => {
    const r = resolveInstruction(
      "Pravin pays for the drinks, split the rest equally",
      DRINKS_CASE,
    );
    expect(r.defaultRuleReason).not.toBe(FALLBACK_DEFAULT_REASON);
    expect(r.promptBlock).toContain("DEFAULT RULE: equal");
  });
});

describe("isInformativeDefaultRule", () => {
  it("exclude is always informative, regardless of reason", () => {
    expect(isInformativeDefaultRule("exclude", FALLBACK_DEFAULT_REASON)).toBe(true);
  });
  it("equal from the uninformative fallback is not informative", () => {
    expect(isInformativeDefaultRule("equal", FALLBACK_DEFAULT_REASON)).toBe(false);
  });
  it("equal from an explicit marker is informative", () => {
    expect(isInformativeDefaultRule("equal", 'equal marker "split the rest"')).toBe(true);
  });
});

describe("resolveInstruction — plain partial item mentions", () => {
  it("resolves a plain partial item mention that isn't a category or exact name", () => {
    // The exact bug that cost two-way-item-split and exclude-unmentioned-items
    // their resolution entirely: "the pizza" vs the full name "PIZZA MARGHERITA".
    const items = [item("PIZZA MARGHERITA"), item("CAESAR SALAD"), item("GARLIC BREAD")];
    const r = resolveInstruction("Split the pizza between Pravin and Aisha", items);
    const pizza = r.references.find((x) => x.itemIndices.includes(0));
    expect(pizza).toBeDefined();
    expect(pizza!.itemIndices).toEqual([0]);
  });

  it("resolves a partial mention AND still detects the exclude rule in the same instruction", () => {
    // exclude-unmentioned-items, reproduced directly: this failed twice
    // over — no item resolution AND the correctly-computed "exclude" was
    // dropped because renderPromptBlock returned "" when references was
    // empty. Both are fixed now.
    const items = [item("BURGER DELUXE"), item("FISH AND CHIPS"), item("MILKSHAKE")];
    const r = resolveInstruction(
      "Just split the burger between Pravin and Sarah, forget about the rest of the table's order",
      items,
    );
    const burger = r.references.find((x) => x.itemIndices.includes(0));
    expect(burger).toBeDefined();
    expect(r.defaultRule).toBe("exclude");
    expect(r.promptBlock).toContain("DEFAULT RULE: exclude");
  });

  it("does not duplicate a reference already covered by an exact-name match", () => {
    // "the teh tarik" contains "the" -> the definite-article scan would
    // also try "teh" and "teh tarik" as candidates; both resolve to the
    // same item the exact-name pass already found. Only one entry should
    // survive for item 2.
    const r = resolveInstruction("Pravin had the teh tarik, split the rest between everyone", DRINKS_CASE);
    const pointingAtTehTarik = r.references.filter((x) => x.itemIndices.length === 1 && x.itemIndices[0] === 2);
    expect(pointingAtTehTarik.length).toBe(1);
  });
});
