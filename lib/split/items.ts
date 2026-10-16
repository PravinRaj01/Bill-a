// A share's `items` is a human-readable string from the engine, e.g. "Nasi Lemak (x1/3), Teh (x2), tax/service",
// or a sentence such as "20% of the bill" / "(nothing assigned)". The receipt shows it as chips.

export interface ItemChip {
  name: string;
  /** How much of the item this person has: "1/3" (an equal split) or "×2" (a weighted share). Omitted for a whole item. */
  share?: string;
}

/** Splits on commas that are not inside brackets, so "Soup (x1/2), Rice" is two items. */
function topLevelParts(text: string): string[] {
  const parts: string[] = [];
  let depth = 0;
  let cur = "";
  for (const ch of text) {
    if (ch === "(") depth++;
    if (ch === ")") depth = Math.max(0, depth - 1);
    if (ch === "," && depth === 0) {
      parts.push(cur);
      cur = "";
    } else cur += ch;
  }
  parts.push(cur);
  return parts.map((p) => p.trim()).filter(Boolean);
}

/** Chips for one share's items; empty when nothing is assigned ("(nothing assigned)", "(not part of this split)"). */
export function parseItems(items: string | undefined | null): ItemChip[] {
  const text = (items ?? "").trim();
  if (!text || (text.startsWith("(") && text.endsWith(")"))) return [];
  return topLevelParts(text).map((part) => {
    const m = /^(.*?)\s*\(x(\d+\/\d+|\d+(?:\.\d+)?)\)$/.exec(part);
    if (!m) return { name: part };
    const [, name, weight] = m;
    if (weight.includes("/")) return { name, share: weight };
    return weight === "1" ? { name } : { name, share: `×${weight}` };
  });
}
