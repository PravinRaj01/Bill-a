import { describe, expect, it } from "vitest";
import { findPersonMentions, levenshtein } from "./people";

describe("levenshtein", () => {
  it.each([["", "", 0], ["a", "", 1], ["kitten", "sitting", 3], ["pravin", "pravn", 1], ["same", "same", 0]] as const)(
    "%j vs %j = %i", (a, b, d) => expect(levenshtein(a, b)).toBe(d),
  );
});

describe("findPersonMentions", () => {
  const people = ["Pravin Raj", "Aisha", "Aisha Tan", "Wei"];

  it("matches whole words case-insensitively, in order of appearance", () => {
    expect(findPersonMentions("wei and pravin raj", people).map((m) => m.person)).toEqual(["Wei", "Pravin Raj"]);
  });

  it("matches a possessive", () => {
    expect(findPersonMentions("Wei's the payer", people).map((m) => m.person)).toEqual(["Wei"]);
  });

  it("matches a first name only when it is unique in the group", () => {
    expect(findPersonMentions("Pravin paid", people).map((m) => m.person)).toEqual(["Pravin Raj"]);
    // "Aisha" is the first name of two people, so a bare "Aisha" matches neither by first name
    expect(findPersonMentions("Aisha paid", people).map((m) => m.person)).toEqual(["Aisha"]);
    expect(findPersonMentions("Aisha Tan paid", people).map((m) => m.person)).toContain("Aisha Tan");
  });

  it("does not match inside another word", () => {
    expect(findPersonMentions("Weird order", people)).toEqual([]);
  });
});
