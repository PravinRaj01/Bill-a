// Every person gets one of 8 colours (--person-1..8 in globals.css), chosen by a stable hash
// of their name so the same person is the same colour on every screen and every device.

export const PERSON_COLOR_COUNT = 8;

export function personColorIndex(name: string): number {
  const key = name.trim().toLowerCase();
  let h = 2166136261; // FNV-1a
  for (let i = 0; i < key.length; i++) {
    h ^= key.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return ((h >>> 0) % PERSON_COLOR_COUNT) + 1;
}

/** CSS colour for a person, e.g. for `style={{ background: personColor(name) }}`. */
export const personColor = (name: string): string => `var(--person-${personColorIndex(name)})`;

/** Two-letter initials: "Pravin Raj" -> "PR", "lakx" -> "LA". */
export function initials(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "?";
  if (parts.length === 1) return Array.from(parts[0]).slice(0, 2).join("").toUpperCase();
  return (Array.from(parts[0])[0] + Array.from(parts[parts.length - 1])[0]).toUpperCase();
}
