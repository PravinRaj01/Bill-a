import { describe, expect, it } from "vitest";

import { formatPhone, normalizePhone, waLink } from "./phone";

describe("normalizePhone", () => {
  it.each([
    ["012-345 6789", "+60123456789"],
    ["0123456789", "+60123456789"],
    ["012 3456789", "+60123456789"],
    ["+60 12-345 6789", "+60123456789"],
    ["+6012 345 6789", "+60123456789"],
    ["(012) 345-6789", "+60123456789"],
    ["+65 9876 5432", "+6598765432"],
    ["  019-876 5432  ", "+60198765432"],
  ])("reads %j as %s", (raw, e164) => {
    expect(normalizePhone(raw)).toBe(e164);
  });

  it.each(["", "   ", "abc", "123", "012-345", "++60123456789", "0000000000", "012-345 678901234567"])("rejects %j", (raw) => {
    expect(normalizePhone(raw)).toBeNull();
  });

  it("uses the given default country for local numbers", () => {
    expect(normalizePhone("9876 5432", "SG")).toBe("+6598765432");
  });
});

describe("formatPhone / waLink", () => {
  it("writes a Malaysian number the local way and others internationally", () => {
    expect(formatPhone("+60123456789")).toBe("012-345 6789");
    expect(formatPhone("+6598765432")).toBe("+65 9876 5432");
    expect(formatPhone("not a number")).toBe("not a number");
  });
  it("builds a wa.me link with digits only and the message encoded", () => {
    expect(waLink("+60123456789", "Hi Riley! You owe RM10.00")).toBe("https://wa.me/60123456789?text=Hi%20Riley!%20You%20owe%20RM10.00");
    expect(waLink("+6598765432", "a&b=c\nd")).toBe("https://wa.me/6598765432?text=a%26b%3Dc%0Ad");
  });
});
