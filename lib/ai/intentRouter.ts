import type { ReceiptItem } from "@/types/domain";
import { AMBIGUITY_THRESHOLD, resolveInstruction, resolveItemReference } from "@/lib/retrieval/resolve";
import { matchCategoryPhrase, normalise } from "@/lib/retrieval/lexicon";
import { findNameProblems, findPersonMentions } from "@/lib/retrieval/people";

// Deterministic intent router. Replaces the "Laya AI" classifier: a few regexes
// on top of resolve.ts, well under a millisecond, and every decision is
// inspectable. It only decides WHAT the user is asking for; the split itself is
// built by the LLM cascade or the fallback parser, and computed by the engine.

export type Intent = "ADD_MEMBERS" | "CALCULATE_SPLIT" | "UNKNOWN";

export interface RouteResult {
  intent: Intent;
  /** For ADD_MEMBERS: the new people to add (not already in the group). */
  newNames: string[];
  reason: string;
}

// "add Wei", "invite Wei and Farah", "include Wei", "+ Wei", "also add Wei", "and also Wei"
const ADD_LEAD = /^\s*(?:please\s+)?(?:(?:can|could)\s+you\s+)?(?:(?:and\s+)?also\s+)?(?:add|invite|include)\b\s*|^\s*\+\s*|^\s*and\s+also\s+/i;
// If the instruction ALSO talks about splitting/paying, it's a split request that
// happens to mention adding someone — don't swallow it as a pure "add" command.
const SPLIT_WORDS =
  /\b(split|share|divide|pay|pays|paying|paid|owe|owes|owing|treat|treats|treating|each|equally|evenly|cover|covers|had|has|except|rest)\b/i;
// Trailing noise after the names: "Wei to the group", "Wei please", "Wei too".
const ADD_TAIL = /\s+(?:to|into|in)\s+(?:the\s+|this\s+|our\s+)?(?:group|bill|split|list|table|people)\b.*$|\s+(?:too|as\s+well|please)\s*$/i;

function parseNewNames(instruction: string, people: string[]): { fresh: string[]; rest: string } | null {
  const lead = ADD_LEAD.exec(instruction);
  if (!lead) return null;
  const rest = instruction.slice(lead[0].length).replace(ADD_TAIL, "").replace(/[.!]+\s*$/, "");
  if (!rest.trim()) return null;

  const names = rest
    .split(/\s*(?:,|&|\band\b)\s*/i)
    .map((n) => n.trim())
    // a name is one to three plain words: rejects sentences like "the pizza to Pravin"
    .filter((n) => /^[A-Za-z][A-Za-z'.-]*(?:\s+[A-Za-z][A-Za-z'.-]*){0,2}$/.test(n))
    .filter((n) => n.length <= 30);
  if (names.length === 0) return null;

  const have = new Set(people.map((p) => normalise(p)));
  const fresh: string[] = [];
  for (const n of names) {
    const key = normalise(n);
    if (!have.has(key) && !fresh.some((f) => normalise(f) === key)) fresh.push(n.replace(/^./, (c) => c.toUpperCase()));
  }
  return { fresh, rest };
}

export function routeIntent(instruction: string, people: string[], items: ReceiptItem[]): RouteResult {
  const text = instruction.trim();
  if (!text) return { intent: "UNKNOWN", newNames: [], reason: "empty instruction" };

  const added = parseNewNames(text, people);
  // Tested on the names part only, so the "…in the split" tail of "include Wei in
  // the split" doesn't make an add command look like a split request.
  if (added && added.fresh.length > 0 && !SPLIT_WORDS.test(added.rest)) {
    return { intent: "ADD_MEMBERS", newNames: added.fresh, reason: "add/invite verb followed by new names" };
  }

  if (SPLIT_WORDS.test(text)) {
    return { intent: "CALCULATE_SPLIT", newNames: [], reason: "split/pay verb" };
  }
  if (findPersonMentions(text, people).length > 0) {
    return { intent: "CALCULATE_SPLIT", newNames: [], reason: "mentions someone in the group" };
  }
  const resolved = resolveInstruction(text, items);
  if (resolved.references.length > 0) {
    return { intent: "CALCULATE_SPLIT", newNames: [], reason: "references items on the receipt" };
  }
  return { intent: "UNKNOWN", newNames: [], reason: "nothing recognisable" };
}

// --- ambiguity: things to ASK about instead of guessing --------------------

export type Ambiguity =
  | { kind: "ambiguous-item"; phrase: string; options: { index: number; name: string }[] }
  | { kind: "unknown-person"; token: string }
  | { kind: "possible-typo"; token: string; suggestion: string };

/**
 * Everything in the instruction the router can't safely act on: an item phrase
 * that ties between several items ("the tea" matching two teas), a word that is
 * probably a typo of someone in the group, or a capitalised name nobody has. The
 * UI turns each into a "did you mean…?" chip BEFORE any LLM call is made.
 */
export function findAmbiguities(
  instruction: string,
  people: string[],
  items: ReceiptItem[],
): Ambiguity[] {
  const out: Ambiguity[] = [];

  // Same "the <1-3 words>" scan resolveInstruction uses, but keeping the ties
  // (confidence below the threshold) that it deliberately drops.
  const tokens = normalise(instruction).split(" ");
  const seen = new Set<string>();
  tokens.forEach((tok, i) => {
    if (tok !== "the") return;
    let tie: { candidate: string; indices: number[] } | null = null;
    for (let len = 1; len <= 3 && i + len < tokens.length; len++) {
      const candidate = tokens.slice(i + 1, i + 1 + len).join(" ");
      if (matchCategoryPhrase(`the ${candidate}`)) return;
      const ref = resolveItemReference(candidate, items);
      if (!ref || ref.itemIndices.length === 0) continue;
      if (ref.confidence >= AMBIGUITY_THRESHOLD) return; // a longer window pins it down ("fried" → "fried rice")
      if (ref.itemIndices.length > 1 && !tie) tie = { candidate, indices: ref.itemIndices };
    }
    if (tie && !seen.has(tie.candidate)) {
      seen.add(tie.candidate);
      out.push({
        kind: "ambiguous-item",
        phrase: tie.candidate,
        options: tie.indices.map((index) => ({ index, name: items[index].name })),
      });
    }
  });

  for (const p of findNameProblems(instruction, people, items.map((it) => it.name))) {
    out.push(
      p.suggestion
        ? { kind: "possible-typo", token: p.token, suggestion: p.suggestion }
        : { kind: "unknown-person", token: p.token },
    );
  }
  return out;
}
