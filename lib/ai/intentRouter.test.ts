import { describe, expect, it } from "vitest";
import { findAmbiguities, routeIntent } from "./intentRouter";

const people = ["Pravin", "Aisha"];
const items = [
  { name: "FRIED RICE", quantity: 1, unitPrice: 900, totalPrice: 900 },
  { name: "FRIED NOODLE", quantity: 1, unitPrice: 800, totalPrice: 800 },
  { name: "TEH TARIK", quantity: 1, unitPrice: 400, totalPrice: 400 },
];

describe("routeIntent", () => {
  it.each([
    ["add Wei", ["Wei"]],
    ["Add Wei and Farah", ["Wei", "Farah"]],
    ["invite Wei", ["Wei"]],
    ["include Wei in the split", ["Wei"]],
    ["+ Wei", ["Wei"]],
    ["and also Wei", ["Wei"]],
    ["please add wei to the group", ["Wei"]],
    ["add Wei, Farah & Sarah", ["Wei", "Farah", "Sarah"]],
  ])("ADD_MEMBERS: %s", (text, names) => {
    expect(routeIntent(text, people, items)).toMatchObject({ intent: "ADD_MEMBERS", newNames: names });
  });

  it("does not treat an already-present name as new", () => {
    expect(routeIntent("add Pravin", people, items).intent).not.toBe("ADD_MEMBERS");
  });

  it("only adds the genuinely new names", () => {
    expect(routeIntent("add Pravin and Wei", people, items)).toMatchObject({ intent: "ADD_MEMBERS", newNames: ["Wei"] });
  });

  it("an add that is also a split request is a split request", () => {
    expect(routeIntent("add Wei and split equally", people, items).intent).toBe("CALCULATE_SPLIT");
    expect(routeIntent("include the drinks for Pravin", people, items).intent).toBe("CALCULATE_SPLIT");
  });

  it.each([
    "Split everything equally",
    "Pravin pays for the drinks",
    "Aisha had the teh tarik",
    "the fried rice is on him",
  ])("CALCULATE_SPLIT: %s", (text) => {
    expect(routeIntent(text, people, items).intent).toBe("CALCULATE_SPLIT");
  });

  it.each(["", "   ", "hello there", "what is the weather"])("UNKNOWN: %j", (text) => {
    expect(routeIntent(text, people, items).intent).toBe("UNKNOWN");
  });
});

describe("findAmbiguities", () => {
  it("flags an item phrase that ties, with the candidates", () => {
    expect(findAmbiguities("Pravin had the fried", people, items)).toContainEqual({
      kind: "ambiguous-item",
      phrase: "fried",
      options: [{ index: 0, name: "FRIED RICE" }, { index: 1, name: "FRIED NOODLE" }],
    });
  });

  it("does not flag a phrase that resolves uniquely", () => {
    expect(findAmbiguities("Pravin had the rice", people, items)).toEqual([]);
  });

  it("suggests the intended person for a typo", () => {
    expect(findAmbiguities("Pravn had the rice", people, items)).toContainEqual({
      kind: "possible-typo", token: "Pravn", suggestion: "Pravin",
    });
    expect(findAmbiguities("pravinn had the rice", people, items)).toContainEqual({
      kind: "possible-typo", token: "pravinn", suggestion: "Pravin",
    });
  });

  it("flags an unknown capitalised name mid-sentence, but not sentence-start or item words", () => {
    expect(findAmbiguities("Pravin and Wei split the Fried Rice", people, items)).toEqual([
      { kind: "unknown-person", token: "Wei" },
    ]);
    expect(findAmbiguities("Split everything equally", people, items)).toEqual([]);
    expect(findAmbiguities("Everyone pays for the Teh Tarik", people, items)).toEqual([]);
  });

  it("never flags exact members, possessives, or contractions", () => {
    expect(findAmbiguities("Pravin's paying, Aisha didn't have the rice", people, items)).toEqual([]);
  });
});
