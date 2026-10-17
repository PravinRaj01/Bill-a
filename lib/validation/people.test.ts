import { describe, expect, it } from "vitest";

import { parseNewPerson, parsePersonPatch } from "./people";

describe("parseNewPerson", () => {
  it("trims the name and stores the number as E.164", () => {
    expect(parseNewPerson({ name: "  Riley ", phone: "012-345 6789" })).toEqual({ ok: true, value: { name: "Riley", phone: "+60123456789" } });
  });
  it("a number is optional", () => {
    expect(parseNewPerson({ name: "Casey" })).toEqual({ ok: true, value: { name: "Casey", phone: null } });
    expect(parseNewPerson({ name: "Casey", phone: "" })).toEqual({ ok: true, value: { name: "Casey", phone: null } });
    expect(parseNewPerson({ name: "Casey", phone: null })).toEqual({ ok: true, value: { name: "Casey", phone: null } });
  });
  it("rejects an empty or long name, the reserved arrow, and a number that isn't one", () => {
    expect(parseNewPerson({ name: "  " })).toEqual({ ok: false, error: "Enter a name" });
    expect(parseNewPerson({ name: "x".repeat(61) }).ok).toBe(false);
    expect(parseNewPerson({ name: "A->B" }).ok).toBe(false);
    expect(parseNewPerson({ name: "Riley", phone: "12" })).toEqual({ ok: false, error: "That doesn't look like a phone number" });
  });
  it("refuses unknown keys, so nothing else (like another user's id) can be written", () => {
    expect(parseNewPerson({ name: "Riley", userId: "someone" }).ok).toBe(false);
    expect(parseNewPerson({ name: "Riley", linkedUserId: "someone" }).ok).toBe(false);
    expect(parseNewPerson("nope").ok).toBe(false);
  });
});

describe("parsePersonPatch", () => {
  it("only includes what was given; null or empty clears the number", () => {
    expect(parsePersonPatch({ name: "Riley T" })).toEqual({ ok: true, value: { name: "Riley T" } });
    expect(parsePersonPatch({ phone: "0123456789" })).toEqual({ ok: true, value: { phone: "+60123456789" } });
    expect(parsePersonPatch({ phone: null })).toEqual({ ok: true, value: { phone: null } });
    expect(parsePersonPatch({ phone: "  " })).toEqual({ ok: true, value: { phone: null } });
    expect(parsePersonPatch({})).toEqual({ ok: true, value: {} });
  });
  it("rejects bad names, bad numbers and unknown keys", () => {
    expect(parsePersonPatch({ name: "" }).ok).toBe(false);
    expect(parsePersonPatch({ phone: "abc" }).ok).toBe(false);
    expect(parsePersonPatch({ id: "x" }).ok).toBe(false);
  });
});
