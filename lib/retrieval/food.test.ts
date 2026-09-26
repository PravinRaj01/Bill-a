import { describe, expect, it } from "vitest";
import { classifyItem } from "./lexicon";
import { resolveInstruction } from "./resolve";

const it_ = (name: string) => ({ name, quantity: 1, unitPrice: 100, totalPrice: 100 });

describe('"the food" means everything that is not a drink', () => {
  const items = [it_("MUSH NOODLES DRY"), it_("STEAM DUMPLINGS"), it_("TEH TARIK"), it_("TIGER BEER"), it_("CHOCOLATE CAKE")];

  it("includes mains, sides, desserts and unclassified items; excludes drinks and alcohol", () => {
    const food = resolveInstruction("Pravin pays for the food", items).references.find((r) => /food/.test(r.phrase));
    expect(food?.itemIndices).toEqual([0, 1, 4]);
  });

  it("on a receipt with no drinks it is every item", () => {
    const two = [it_("MUSH NOODLES DRY"), it_("STEAM DUMPLINGS")];
    const food = resolveInstruction("the food", two).references[0];
    expect(food.itemIndices).toEqual([0, 1]);
  });

  it("'the meal' works the same way", () => {
    expect(resolveInstruction("Aisha pays the meal", items).references.some((r) => r.itemIndices.join() === "0,1,4")).toBe(true);
  });

  it("other categories are unchanged: 'the drinks' is still just the drinks", () => {
    const drinks = resolveInstruction("Pravin pays for the drinks", items).references.find((r) => /drink/.test(r.phrase));
    expect(drinks?.itemIndices).toEqual([2]);
  });

  it("'the mains' is still the mains only", () => {
    const mains = resolveInstruction("Pravin pays for the mains", items).references.find((r) => /mains/.test(r.phrase));
    expect(mains?.itemIndices).toEqual([0]);
  });
});

describe("plural item names match their singular keywords", () => {
  it.each([
    ["STEAM DUMPLINGS", "sides"],
    ["FRIED NOODLES", "mains"],
    ["SPRING ROLLS", "sides"],
    ["ICED LEMON TEAS", "drinks"],
  ])("%s -> %s", (name, cat) => {
    expect(classifyItem(name)[0]?.categoryId).toBe(cat);
  });
});
