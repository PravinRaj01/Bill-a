import { describe, expect, it } from "vitest";

import { avatarDataUri, avatarSeed, avatarsFor, cachedAvatar } from "./avatar";

describe("avatars", () => {
  it("makes an SVG data URI", async () => {
    const uri = await avatarDataUri("Alexis");
    expect(uri.startsWith("data:image/svg+xml")).toBe(true);
    expect(uri.length).toBeGreaterThan(500);
  });

  it("is the same face for the same person, however it was typed", async () => {
    expect(avatarSeed("  Riley ")).toBe("riley");
    expect(await avatarDataUri("Riley")).toBe(await avatarDataUri("  RILEY "));
  });

  it("gives different people different faces", async () => {
    expect(await avatarDataUri("Riley")).not.toBe(await avatarDataUri("Casey"));
  });

  it("is deterministic and cached", async () => {
    const a = await avatarDataUri("Sam");
    expect(cachedAvatar("sam")).toBe(a);
    expect(cachedAvatar("someone never asked for")).toBeUndefined();
    expect(await avatarDataUri("Sam")).toBe(a);
  });

  it("makes a map for a whole group, keyed by the exact name", async () => {
    const m = await avatarsFor(["Alexis", "Riley"]);
    expect(Object.keys(m)).toEqual(["Alexis", "Riley"]);
    expect(m.Alexis).toBe(await avatarDataUri("Alexis"));
  });
});
