import type { ProviderId } from "./providers/types";

// The user's own API keys, kept in THIS browser's localStorage and nowhere else.
//
//  - They are never sent to our servers, put in a URL, logged, or synced. The only
//    requests that carry one go to api.groq.com and generativelanguage.googleapis.com
//    (next.config.ts pins the CSP `connect-src` to exactly those origins).
//  - localStorage is readable by any script on the page, so an XSS bug could read a
//    key. The CSP is the mitigation (a stolen key can't be sent to an attacker's
//    server), and a leaked free-tier key's blast radius is limited to that key's quota.
//  - Every access is try/catch'd: private windows and blocked storage throw. Then keys
//    live in memory for the tab's lifetime instead, and the UI keeps working.

const STORAGE_KEY = "billa.ai.keys.v1";

export type Keys = Partial<Record<ProviderId, string>>;

let memory: Keys = {};
const listeners = new Set<() => void>();

function readStorage(): Keys | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw === null) return {};
    const parsed = JSON.parse(raw) as unknown;
    if (typeof parsed !== "object" || parsed === null) return {};
    const out: Keys = {};
    for (const id of ["groq", "gemini"] as const) {
      const v = (parsed as Record<string, unknown>)[id];
      if (typeof v === "string" && v.trim()) out[id] = v.trim();
    }
    return out;
  } catch {
    return null; // storage unavailable
  }
}

function writeStorage(keys: Keys): boolean {
  try {
    if (Object.keys(keys).length === 0) localStorage.removeItem(STORAGE_KEY);
    else localStorage.setItem(STORAGE_KEY, JSON.stringify(keys));
    return true;
  } catch {
    return false;
  }
}

export function getKeys(): Keys {
  return { ...(readStorage() ?? memory) };
}

/** Saves (or, with an empty string, removes) a key. Returns whether it persisted beyond this tab. */
export function setKey(provider: ProviderId, key: string): { persisted: boolean } {
  const next = getKeys();
  const clean = key.trim();
  if (clean) next[provider] = clean;
  else delete next[provider];
  memory = next;
  const persisted = writeStorage(next);
  listeners.forEach((l) => l());
  return { persisted };
}

export function removeKey(provider: ProviderId) {
  return setKey(provider, "");
}

export function hasAnyKey(): boolean {
  return Object.keys(getKeys()).length > 0;
}

/** Subscribe to key changes (this tab, and other tabs via the storage event). */
export function onKeysChanged(cb: () => void): () => void {
  listeners.add(cb);
  const onStorage = (e: StorageEvent) => {
    if (e.key === STORAGE_KEY) cb();
  };
  if (typeof window !== "undefined") window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(cb);
    if (typeof window !== "undefined") window.removeEventListener("storage", onStorage);
  };
}

/** Never show a whole key back to the user (or in a screenshot): gsk_…a1b2. */
export function maskKey(key: string): string {
  return key.length <= 8 ? "••••" : `${key.slice(0, 4)}…${key.slice(-4)}`;
}

/** Format sanity check only — a wrong prefix warns, it doesn't block (formats change). */
export function looksLikeKey(provider: ProviderId, key: string): boolean {
  const k = key.trim();
  return provider === "groq" ? /^gsk_[A-Za-z0-9]{20,}$/.test(k) : /^AIza[\w-]{30,}$/.test(k);
}
