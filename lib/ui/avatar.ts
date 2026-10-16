// Illustrated avatars (DiceBear "notionists": hand-drawn line portraits) generated from a person's name, so
// the same person always has the same face on every screen and device, with nothing uploaded or stored.
//
// The style pack is a few hundred KB, so it is loaded on demand with a dynamic import: screens render
// initials first and swap the portrait in when it arrives. Results are cached per person.

import { personColorIndex } from "@/lib/ui/person-color";

// Notionists draws dark line art, which vanishes on the app's dark surface, so each portrait gets a soft
// pastel background. The eight pastels line up with the eight person colours (--person-1..8), so a
// person's ring and face agree.
const PASTELS = ["ffd6d1", "ffe3c2", "eef3b8", "c9f2d6", "bfe9f2", "cdd8ff", "e6d3fb", "ffd0e4"];

/** The same person however it was typed: "RILEY " and "riley" are one person (as in the balances). */
export const avatarSeed = (name: string): string => name.trim().toLowerCase();

const cache = new Map<string, string>();
const pending = new Map<string, Promise<string>>();

/** The portrait if it has already been made, for rendering without a flash. */
export const cachedAvatar = (name: string): string | undefined => cache.get(avatarSeed(name));

/** A `data:image/svg+xml` URI for this person's portrait. */
export function avatarDataUri(name: string): Promise<string> {
  const seed = avatarSeed(name);
  const hit = cache.get(seed);
  if (hit) return Promise.resolve(hit);
  const inflight = pending.get(seed);
  if (inflight) return inflight;
  const made = (async () => {
    const [{ createAvatar }, style] = await Promise.all([import("@dicebear/core"), import("@dicebear/notionists")]);
    const uri = createAvatar(style, {
      seed,
      backgroundColor: [PASTELS[personColorIndex(name) - 1]],
      scale: 110,
    }).toDataUri();
    cache.set(seed, uri);
    pending.delete(seed);
    return uri;
  })();
  pending.set(seed, made);
  return made;
}

/** Portraits for everyone, by exact name (for the receipt image, which must have them before it is captured). */
export async function avatarsFor(names: string[]): Promise<Record<string, string>> {
  const entries = await Promise.all(names.map(async (n) => [n, await avatarDataUri(n)] as const));
  return Object.fromEntries(entries);
}
