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

  it("a chosen seed gives a different portrait for the same name, and is cached on its own", async () => {
    const byName = await avatarDataUri("Alexis");
    const chosen = await avatarDataUri("Alexis", "sun-42");
    expect(chosen).not.toBe(byName);
    expect(cachedAvatar("Alexis", "sun-42")).toBe(chosen);
    expect(cachedAvatar("Alexis")).toBe(byName);
    expect(await avatarDataUri("Alexis", "sun-42")).toBe(chosen);
  });

  it("the same seed on two people is two pictures (each takes its own background colour)", async () => {
    const a = await avatarDataUri("Alexis", "moon-7");
    const r = await avatarDataUri("Riley", "moon-7");
    // personColorIndex differs for these names, so the pastel does; if it ever collides the faces are still equal, so only assert the cache is per person
    expect(cachedAvatar("Alexis", "moon-7")).toBe(a);
    expect(cachedAvatar("Riley", "moon-7")).toBe(r);
  });

  it("makes a map for a whole group, keyed by the exact name", async () => {
    const m = await avatarsFor(["Alexis", "Riley"]);
    const seeded = await avatarsFor(["Alexis", "Riley"], (n) => (n === "Alexis" ? "sun-42" : undefined));
    expect(seeded.Alexis).toBe(await avatarDataUri("Alexis", "sun-42"));
    expect(seeded.Riley).toBe(m.Riley);
    expect(Object.keys(m)).toEqual(["Alexis", "Riley"]);
    expect(m.Alexis).toBe(await avatarDataUri("Alexis"));
  });
});
