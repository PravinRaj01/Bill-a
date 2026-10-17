import { describe, expect, it } from "vitest";

import { parseProfileInput, validTimeZone } from "./profile";

describe("parseProfileInput", () => {
  it("lower-cases and trims a username", () => {
    expect(parseProfileInput({ username: "  Alexis_01 " })).toEqual({ ok: true, value: { username: "alexis_01" } });
  });
  it("accepts clearing (null) and leaving alone (absent)", () => {
    expect(parseProfileInput({ username: null, avatarSeed: null })).toEqual({ ok: true, value: { username: null, avatarSeed: null } });
    expect(parseProfileInput({})).toEqual({ ok: true, value: {} });
  });
  it.each(["ab", "a".repeat(21), "has space", "dash-ed", "émile", "al.ex", ""])("rejects the format %j", (u) => {
    expect(parseProfileInput({ username: u }).ok).toBe(false);
  });
  it.each(["admin", "Billa", "SUPPORT", "you", "login"])("rejects the reserved name %j", (u) => {
    expect(parseProfileInput({ username: u })).toEqual({ ok: false, error: "That username isn't available" });
  });
  it("bounds the portrait seed to plain characters", () => {
    expect(parseProfileInput({ avatarSeed: "sun-42" }).ok).toBe(true);
    expect(parseProfileInput({ avatarSeed: "x".repeat(41) }).ok).toBe(false);
    expect(parseProfileInput({ avatarSeed: "<script>" }).ok).toBe(false);
    expect(parseProfileInput({ avatarSeed: "  " }).ok).toBe(false);
  });
  it("refuses unknown keys, so nothing else can be written through this action", () => {
    expect(parseProfileInput({ username: "alexis", email: "x@y.z" }).ok).toBe(false);
    expect(parseProfileInput({ id: "someone-else" }).ok).toBe(false);
    expect(parseProfileInput("nope").ok).toBe(false);
  });
});

describe("validTimeZone", () => {
  it("accepts real zones and rejects junk", () => {
    expect(validTimeZone("Asia/Kuala_Lumpur")).toBe("Asia/Kuala_Lumpur");
    expect(validTimeZone("UTC")).toBe("UTC");
    expect(validTimeZone("Mars/Olympus")).toBeNull();
    expect(validTimeZone("x; drop table")).toBeNull();
    expect(validTimeZone(42)).toBeNull();
    expect(validTimeZone("A".repeat(65))).toBeNull();
  });
});
