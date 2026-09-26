import { describe, expect, it } from "vitest";
import { burnVerifyTime, hashPassword, verifyPassword } from "./password";

describe("password hashing", () => {
  it("hashes to argon2id and verifies the right password", async () => {
    const h = await hashPassword("correct horse battery staple");
    expect(h.startsWith("$argon2id$")).toBe(true);
    expect(await verifyPassword(h, "correct horse battery staple")).toBe(true);
  });

  it("rejects a wrong password", async () => {
    const h = await hashPassword("hunter2hunter2");
    expect(await verifyPassword(h, "hunter2hunter3")).toBe(false);
    expect(await verifyPassword(h, "")).toBe(false);
  });

  it("salts: the same password hashes differently each time", async () => {
    const a = await hashPassword("same-password-123");
    const b = await hashPassword("same-password-123");
    expect(a).not.toBe(b);
    expect(await verifyPassword(a, "same-password-123")).toBe(true);
    expect(await verifyPassword(b, "same-password-123")).toBe(true);
  });

  it("treats a malformed stored hash as a wrong password instead of throwing", async () => {
    expect(await verifyPassword("not-a-real-hash", "whatever")).toBe(false);
    expect(await verifyPassword("", "whatever")).toBe(false);
  });

  it("burnVerifyTime resolves without throwing (used for unknown emails)", async () => {
    await expect(burnVerifyTime("anything")).resolves.toBeUndefined();
  });
});
