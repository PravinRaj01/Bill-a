import { describe, expect, it } from "vitest";

import { billMe, buildParticipants, isHosted, meNameFor, nameProblem, participantProblem, resolveSelfRefs, samePerson } from "./participants";
import type { Participant } from "@/types/domain";

describe("meNameFor / samePerson", () => {
  it("uses the first word of the account name, else Me", () => {
    expect(meNameFor("Alexis Reed")).toBe("Alexis");
    expect(meNameFor("  riley ")).toBe("riley");
    expect(meNameFor(null)).toBe("Me");
    expect(meNameFor("")).toBe("Me");
  });
  it("ignores case and spaces", () => {
    expect(samePerson(" Alexis ", "alexis")).toBe(true);
    expect(samePerson("Alex", "Alexis")).toBe(false);
  });
});

describe("buildParticipants / billMe / isHosted", () => {
  it("marks the user once, matching case-insensitively, and keeps the spelling", () => {
    expect(buildParticipants(["Alexis", "Riley"], "alexis")).toEqual([{ name: "Alexis", me: true }, { name: "Riley" }]);
    expect(buildParticipants(["Alexis", "alexis"], "Alexis").filter((p) => p.me)).toHaveLength(1);
  });
  it("marks nobody when split for others", () => {
    expect(buildParticipants(["Riley", "Casey"], null).some((p) => p.me)).toBe(false);
  });
  it("billMe reads the participants, falling back to the old settlement.me", () => {
    const parts: Participant[] = [{ name: "Alexis", me: true }, { name: "Riley" }];
    expect(billMe({ participants: parts })).toBe("Alexis");
    expect(billMe({ settlement: { me: "Riley", paidBy: [], paid: {} } })).toBe("Riley");
    expect(billMe({ participants: [{ name: "Riley" }], settlement: { me: "Riley", paidBy: [], paid: {} } })).toBeUndefined();
    expect(billMe({})).toBeUndefined();
  });
  it("isHosted needs participants and no me; old bills are never hosted", () => {
    expect(isHosted({ participants: [{ name: "Riley" }] })).toBe(true);
    expect(isHosted({ participants: [{ name: "Alexis", me: true }] })).toBe(false);
    expect(isHosted({})).toBe(false);
  });
});

describe("nameProblem", () => {
  it("accepts a new name and rejects empty, duplicate, you and arrows", () => {
    expect(nameProblem("Sam", ["Alexis"], "Alexis")).toBeNull();
    expect(nameProblem("   ", [], null)).toBe("empty");
    expect(nameProblem("riley", ["Riley"], "Alexis")).toBe("duplicate");
    expect(nameProblem("alexis", ["Alexis"], "Alexis")).toBe("you");
    expect(nameProblem("A->B", [], null)).toBe("arrow");
  });
});

describe("participantProblem", () => {
  const people = ["Alexis", "Riley"];
  const base = { people, participants: [{ name: "Alexis", me: true as const }, { name: "Riley" }], settlement: { me: "Alexis", paidBy: [], paid: {} } };
  it("passes a consistent bill, and old bills without participants", () => {
    expect(participantProblem(base)).toBeNull();
    expect(participantProblem({ people, settlement: { me: "Whoever", paidBy: [], paid: {} } })).toBeNull();
  });
  it("passes a hosted bill with no settlement.me", () => {
    expect(participantProblem({ people, participants: [{ name: "Alexis" }, { name: "Riley" }] })).toBeNull();
  });
  it("rejects mismatches, duplicates, arrows, two mes and a linked me", () => {
    expect(participantProblem({ ...base, people: ["Alexis", "Sam"] })).toMatch(/match/);
    expect(participantProblem({ people: ["Alex", "alex"], participants: [{ name: "Alex" }, { name: "alex" }] })).toMatch(/duplicate/);
    expect(participantProblem({ people: ["A->B"], participants: [{ name: "A->B" }] })).toMatch(/->/);
    expect(participantProblem({ ...base, participants: [{ name: "Alexis", me: true }, { name: "Riley", me: true }] })).toMatch(/more than one/);
    expect(participantProblem({ ...base, participants: [{ name: "Alexis", me: true, userId: "u1" }, { name: "Riley" }] })).toMatch(/link/);
  });
  it("rejects a settlement.me that disagrees", () => {
    expect(participantProblem({ ...base, settlement: { me: "Riley", paidBy: [], paid: {} } })).toMatch(/disagrees/);
    expect(participantProblem({ people, participants: [{ name: "Alexis" }, { name: "Riley" }], settlement: { me: "Alexis", paidBy: [], paid: {} } })).toMatch(/disagrees/);
  });
});

describe("resolveSelfRefs", () => {
  const people = ["Alexis", "Riley"];
  it("swaps I / me / myself for the Me name", () => {
    expect(resolveSelfRefs("I didn't have the rice", "Alexis", people)).toBe("Alexis didn't have the rice");
    expect(resolveSelfRefs("split between me and Riley", "Alexis", people)).toBe("split between Alexis and Riley");
    expect(resolveSelfRefs("i had the cake", "Alexis", people)).toBe("Alexis had the cake");
  });
  it("leaves contractions, other words and non-members alone", () => {
    expect(resolveSelfRefs("I'm not eating", "Alexis", people)).toBe("I'm not eating");
    expect(resolveSelfRefs("Riley had the iced tea", "Alexis", people)).toBe("Riley had the iced tea");
    expect(resolveSelfRefs("I had it", "Alexis", ["Riley"])).toBe("I had it");
    expect(resolveSelfRefs("I had it", null, people)).toBe("I had it");
    expect(resolveSelfRefs("me too", "Me", ["Me"])).toBe("me too");
  });
});
