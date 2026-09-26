import { beforeEach, describe, expect, it, vi } from "vitest";
import { getKeys, hasAnyKey, looksLikeKey, maskKey, onKeysChanged, removeKey, setKey } from "./keyStore";

function installStorage(impl: Partial<Storage> & { data?: Map<string, string> }) {
  const data = impl.data ?? new Map<string, string>();
  vi.stubGlobal("localStorage", {
    getItem: (k: string) => (data.has(k) ? data.get(k)! : null),
    setItem: (k: string, v: string) => void data.set(k, v),
    removeItem: (k: string) => void data.delete(k),
    ...impl,
  });
  return data;
}

beforeEach(() => {
  vi.unstubAllGlobals();
  installStorage({});
  removeKey("groq");
  removeKey("gemini");
});

describe("keyStore", () => {
  it("stores keys per provider in localStorage only", () => {
    const data = installStorage({});
    setKey("groq", "  gsk_abc  ");
    expect(getKeys()).toEqual({ groq: "gsk_abc" });
    expect(data.size).toBe(1);
    expect(hasAnyKey()).toBe(true);
    setKey("gemini", "AIza123");
    expect(getKeys()).toEqual({ groq: "gsk_abc", gemini: "AIza123" });
    removeKey("groq");
    expect(getKeys()).toEqual({ gemini: "AIza123" });
    removeKey("gemini");
    expect(hasAnyKey()).toBe(false);
    expect(data.size).toBe(0); // the storage entry itself is removed, not left as "{}"
  });

  it("an empty string removes the key", () => {
    setKey("groq", "gsk_abc");
    setKey("groq", "   ");
    expect(getKeys()).toEqual({});
  });

  it("survives blocked storage (private window): keys live in memory and it says so", () => {
    installStorage({
      getItem: () => { throw new DOMException("denied", "SecurityError"); },
      setItem: () => { throw new DOMException("denied", "SecurityError"); },
      removeItem: () => { throw new DOMException("denied", "SecurityError"); },
    });
    expect(() => getKeys()).not.toThrow();
    expect(setKey("groq", "gsk_mem")).toEqual({ persisted: false });
    expect(getKeys()).toEqual({ groq: "gsk_mem" });
  });

  it("ignores a corrupted storage value instead of crashing", () => {
    const data = installStorage({});
    data.set("billa.ai.keys.v1", "{not json");
    expect(getKeys()).toEqual({});
    data.set("billa.ai.keys.v1", JSON.stringify({ groq: 5, gemini: "AIzaOK", evil: "x" }));
    expect(getKeys()).toEqual({ gemini: "AIzaOK" }); // wrong types and unknown providers dropped
  });

  it("notifies subscribers when keys change", () => {
    const cb = vi.fn();
    const off = onKeysChanged(cb);
    setKey("groq", "gsk_abc");
    expect(cb).toHaveBeenCalledTimes(1);
    off();
    setKey("groq", "gsk_def");
    expect(cb).toHaveBeenCalledTimes(1);
  });

  it("masks keys so a whole one is never shown", () => {
    expect(maskKey("gsk_abcdefghijklmnop1234")).toBe("gsk_…1234");
    expect(maskKey("short")).toBe("••••");
    expect(maskKey("gsk_abcdefghijklmnop1234")).not.toContain("abcdefgh");
  });

  it("checks key shapes (as a hint only)", () => {
    expect(looksLikeKey("groq", "gsk_" + "a".repeat(40))).toBe(true);
    expect(looksLikeKey("groq", "AIza" + "a".repeat(35))).toBe(false);
    expect(looksLikeKey("gemini", "AIza" + "a".repeat(35))).toBe(true);
    expect(looksLikeKey("gemini", "gsk_" + "a".repeat(40))).toBe(false);
    expect(looksLikeKey("groq", "")).toBe(false);
  });
});
