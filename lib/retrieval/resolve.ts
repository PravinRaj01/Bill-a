import type { ReceiptItem } from "@/types/domain";
import { classifyItem, matchCategoryPhrase, normalise, CATEGORIES } from "./lexicon";

// The resolution layer — plan §5.1b.
//
// Takes an instruction plus the item list and works out WHICH ITEMS each
// phrase refers to, so the LLM never has to search. The Phase 1 bake-off
// showed every model from 1.5B to 3.8B failing at exactly this, while
// passing the single-clause form that remains once it's done.
//
// Everything in this file is deterministic and synchronous. Tiers 2 and 3
// of the cascade (embeddings, per-item binary LLM calls) are async and
// plug in via `resolveWithFallback` — but the aim is for tier 1 to carry
// most of the load, because it's free and fully inspectable.

export type ResolutionVia =
  | "exact-name"
  | "partial-name"
  | "category"
  | "complement"
  | "embedding"
  | "llm";

export interface ResolvedReference {
  /** The phrase as it appeared in the instruction. */
  phrase: string;
  itemIndices: number[];
  /** 0-1. Below AMBIGUITY_THRESHOLD the caller should escalate to tier 2/3. */
  confidence: number;
  via: ResolutionVia;
  /** True when the phrase is introduced by "except"/"apart from"/etc. */
  isException?: boolean;
}

export interface ResolutionResult {
  references: ResolvedReference[];
  defaultRule: "equal" | "exclude";
  defaultRuleReason: string;
  /** Ready to splice into the prompt; empty string when nothing resolved. */
  promptBlock: string;
}

export const AMBIGUITY_THRESHOLD = 0.5;

// --- defaultRule detection -------------------------------------------------
//
// The bake-off showed every model flip-flopping between "equal" and
// "exclude" inconsistently, sometimes within the same run. It's a binary
// classification over structural markers, so code should own it.
//
// Note "just"/"only" are deliberately NOT bare markers: "which is just
// for Sarah" means an item belongs exclusively to a person, not that the
// rest of the bill is excluded. They only count immediately before a
// split verb.

const EXCLUDE_MARKERS: Array<{ re: RegExp; reason: string }> = [
  { re: /\b(just|only)\s+(split|divide|share)\b/, reason: '"just/only split ..."' },
  { re: /\bforget\s+(about\s+)?(the\s+)?rest\b/, reason: '"forget the rest"' },
  { re: /\bignore\s+(the\s+)?(rest|others|other\s+items|everything\s+else)\b/, reason: '"ignore the rest"' },
  { re: /\bnothing\s+else\b/, reason: '"nothing else"' },
  { re: /\bdon'?t\s+(include|count|add)\s+(the\s+)?(rest|others)\b/, reason: '"don\'t include the rest"' },
  { re: /\bleave\s+out\s+the\s+rest\b/, reason: '"leave out the rest"' },
];

const EQUAL_MARKERS: Array<{ re: RegExp; reason: string }> = [
  { re: /\b(split|share|divide)\s+(up\s+)?the\s+rest\b/, reason: '"split the rest"' },
  { re: /\bthe\s+rest\s+(is\s+|are\s+|gets?\s+)?(split|shared|divided|equally)\b/, reason: '"the rest ... equally"' },
  { re: /\brest\s+(between|among)\b/, reason: '"the rest between everyone"' },
  { re: /\beveryone\s+(else\s+)?(splits?|shares?|pays?)\b/, reason: '"everyone splits"' },
  { re: /\b(everything|anything)\s+else\s+(is\s+)?(split|shared|equally)\b/, reason: '"everything else equally"' },
  { re: /\bremainder\b/, reason: '"the remainder"' },
  { re: /\bsplit\s+(it\s+|everything\s+|the\s+bill\s+)?equally\b/, reason: '"split equally"' },
];

export function detectDefaultRule(instruction: string): {
  defaultRule: "equal" | "exclude";
  reason: string;
} {
  const text = normalise(instruction);

  for (const m of EXCLUDE_MARKERS) {
    if (m.re.test(text)) return { defaultRule: "exclude", reason: `exclude marker ${m.reason}` };
  }
  for (const m of EQUAL_MARKERS) {
    if (m.re.test(text)) return { defaultRule: "equal", reason: `equal marker ${m.reason}` };
  }
  // Default to "equal": covering every item is the safer failure mode —
  // "exclude" can silently leave part of the bill unpaid by anyone.
  return { defaultRule: "equal", reason: FALLBACK_DEFAULT_REASON };
}

/** Sentinel for "we didn't find a marker, we're just guessing equal" — see isInformativeDefaultRule. */
export const FALLBACK_DEFAULT_REASON = "no marker found, defaulting to equal";

/**
 * True when defaultRule was actually derived from something in the text
 * (an explicit "exclude"/"equal" marker), false when it's just the code's
 * own uninformative fallback guess.
 *
 * This distinction matters because of a regression found in the
 * resolution-layer bake-off: injecting "DEFAULT RULE: equal" into the
 * prompt for a case where nothing needed resolving (e.g. "Split
 * everything equally", no category, no exception) added tokens that
 * changed Llama-3.2-1B's greedy decode path to a WRONG answer on a case
 * it previously got right — pure noise, since "equal" is already the
 * model's own stated fallback per the system prompt. `exclude` is never
 * suppressed: that's the genuinely new information that fixed
 * exclude-unmentioned-items in the first place.
 */
export function isInformativeDefaultRule(defaultRule: "equal" | "exclude", reason: string): boolean {
  return defaultRule === "exclude" || reason !== FALLBACK_DEFAULT_REASON;
}

// --- exception detection ---------------------------------------------------

const EXCEPTION_MARKERS = [
  /\bexcept\s+(for\s+)?/,
  /\bapart\s+from\s+/,
  /\bother\s+than\s+/,
  /\ball\s+but\s+/,
  /\bbut\s+not\s+/,
  /\bexcluding\s+/,
];

/** Returns the text following an exception marker, if any. */
export function extractExceptionClause(instruction: string): string | null {
  const text = normalise(instruction);
  for (const re of EXCEPTION_MARKERS) {
    const m = re.exec(text);
    if (m) return text.slice(m.index + m[0].length).trim();
  }
  return null;
}

// --- item reference resolution (tier 1) ------------------------------------

/**
 * Resolves a phrase to item indices using only deterministic signals:
 * exact item-name containment, then token overlap, then the category
 * lexicon. Returns null when nothing matches, so the caller can escalate.
 */
export function resolveItemReference(
  phrase: string,
  items: ReceiptItem[],
): ResolvedReference | null {
  const needle = normalise(phrase);
  if (!needle) return null;

  // 1. Exact-ish: the phrase contains the item's full name (or vice versa).
  const exact: number[] = [];
  items.forEach((item, i) => {
    const name = normalise(item.name);
    if (needle.includes(name) || name === needle) exact.push(i);
  });
  if (exact.length > 0) {
    return { phrase, itemIndices: exact, confidence: 1, via: "exact-name" };
  }

  // 2. Category phrase — "the drinks" -> every item classified as a drink.
  const catId = matchCategoryPhrase(needle);
  if (catId) {
    // "the food" (and "the meal") in everyday speech means everything that isn't a drink —
    // NOT just the mains: "Pravin pays for the food, Wifey pays the tax" must include the
    // dumplings and the dessert. Unclassified items count as food.
    const meansAllFood = catId === "mains" && /(?:^|[^a-z])(?:food|meals?)(?:[^a-z]|$)/.test(needle);
    const hits: number[] = [];
    items.forEach((item, i) => {
      const top = classifyItem(item.name)[0]?.categoryId;
      if (meansAllFood ? top !== "drinks" && top !== "alcohol" : top === catId) hits.push(i);
    });
    if (hits.length > 0) {
      return { phrase, itemIndices: hits, confidence: 0.9, via: "category" };
    }
    // The category was recognised but matched no item — a real signal that
    // tier 2/3 should take over rather than us returning an empty set.
    return { phrase, itemIndices: [], confidence: 0, via: "category" };
  }

  // 3. Token overlap — "the teh tarik" -> "TEH TARIK"; "the tea" -> both
  //    tea items (ambiguous, reflected in the confidence).
  const needleTokens = new Set(needle.split(" ").filter((t) => t.length > 2));
  if (needleTokens.size === 0) return null;

  const scored = items
    .map((item, i) => {
      const nameTokens = normalise(item.name).split(" ");
      const overlap = nameTokens.filter((t) => needleTokens.has(t)).length;
      return { i, overlap, nameLen: nameTokens.length };
    })
    .filter((s) => s.overlap > 0)
    .sort((a, b) => b.overlap - a.overlap);

  if (scored.length === 0) return null;

  const topOverlap = scored[0].overlap;
  const winners = scored.filter((s) => s.overlap === topOverlap);
  // Several items match equally well -> genuinely ambiguous. Report low
  // confidence rather than guessing; the caller escalates.
  const confidence = winners.length === 1 ? 0.75 : 0.4;

  return {
    phrase,
    itemIndices: winners.map((w) => w.i),
    confidence,
    via: "partial-name",
  };
}

// --- set operations (never asked of the model) -----------------------------

export function complement(indices: number[], itemCount: number): number[] {
  const excluded = new Set(indices);
  const out: number[] = [];
  for (let i = 0; i < itemCount; i++) if (!excluded.has(i)) out.push(i);
  return out;
}

// --- orchestration ---------------------------------------------------------

const COMPLEMENT_PHRASES = [
  "the rest", "everything else", "anything else", "the others",
  "the remainder", "whatever's left", "whats left", "the other items",
];

/**
 * Full tier-1 resolution. Scans the instruction for category phrases,
 * exact item names, and partial item mentions ("the pizza"), resolves
 * each to indices, detects exceptions and "the rest", and computes
 * complements in code.
 *
 * Item references fail open: anything unresolved is simply absent from
 * the result, so the model sees what it sees today and is never worse
 * off than the baseline. `defaultRule` does NOT fail open the same way —
 * it is always computed and always surfaced in `promptBlock`, even when
 * zero item references resolved, because it's a separate, independent
 * decision (see renderPromptBlock's doc comment for the bug this fixed).
 */
export function resolveInstruction(
  instruction: string,
  items: ReceiptItem[],
): ResolutionResult {
  const text = normalise(instruction);
  const references: ResolvedReference[] = [];
  const seen = new Set<string>();

  // Category phrases mentioned anywhere in the instruction.
  for (const cat of CATEGORIES) {
    for (const qp of cat.queryPhrases) {
      const n = normalise(qp);
      if (!text.includes(n) || seen.has(n)) continue;
      const ref = resolveItemReference(n, items);
      if (ref && ref.itemIndices.length > 0) {
        references.push(ref);
        seen.add(n);
      }
    }
  }

  // Item names mentioned directly ("Pravin had the teh tarik").
  items.forEach((item, i) => {
    const name = normalise(item.name);
    if (text.includes(name) && !seen.has(name)) {
      references.push({ phrase: name, itemIndices: [i], confidence: 1, via: "exact-name" });
      seen.add(name);
    }
  });

  // Bug found in the resolution-layer bake-off run: the two passes above
  // only catch a FULL category phrase or a FULL exact item name. A plain
  // partial mention — "the pizza" against "PIZZA MARGHERITA", "the
  // burger" against "BURGER DELUXE" — matched neither, so it silently
  // got NO resolution at all. That single gap cost two of the five cases
  // this layer targets (two-way-item-split, exclude-unmentioned-items)
  // their entire benefit. Scan every "the <1-3 words>" span and run it
  // through resolveItemReference's token-overlap tier, shortest window
  // first. Anything a stronger pass above already fully explains is
  // skipped, so "the teh tarik" doesn't also produce a redundant "teh" ->
  // [2] entry alongside the exact "teh tarik" -> [2] one.
  const exactlyClaimed = new Set(references.filter((r) => r.confidence === 1).flatMap((r) => r.itemIndices));
  const tokens = text.split(" ");
  tokens.forEach((tok, i) => {
    if (tok !== "the") return;
    for (let len = 1; len <= 3 && i + len < tokens.length; len++) {
      const candidate = tokens.slice(i + 1, i + 1 + len).join(" ");
      if (seen.has(candidate)) break;
      if (matchCategoryPhrase(`the ${candidate}`)) break; // category pass already owns this
      if (COMPLEMENT_PHRASES.includes(`the ${candidate}`)) break; // complement pass owns this

      const ref = resolveItemReference(candidate, items);
      if (ref && ref.itemIndices.length > 0 && ref.confidence >= AMBIGUITY_THRESHOLD) {
        const alreadyFullyCovered = ref.itemIndices.every((idx) => exactlyClaimed.has(idx));
        if (!alreadyFullyCovered) {
          references.push(ref);
          seen.add(candidate);
          if (ref.confidence === 1) ref.itemIndices.forEach((idx) => exactlyClaimed.add(idx));
        }
        break; // stop growing the window once the shortest match is found
      }
    }
  });

  // Mark references introduced by "except"/"apart from"/... — these get
  // their own explicit assignment rather than following defaultRule.
  const exceptionClause = extractExceptionClause(instruction);
  if (exceptionClause) {
    for (const ref of references) {
      if (exceptionClause.includes(ref.phrase)) ref.isException = true;
    }
  }

  // "the rest" / "everything else" -> complement of everything referenced.
  const mentionedComplement = COMPLEMENT_PHRASES.find((p) => text.includes(normalise(p)));
  if (mentionedComplement) {
    const claimed = new Set(references.flatMap((r) => r.itemIndices));
    const rest = complement([...claimed], items.length);
    if (rest.length > 0 && claimed.size > 0) {
      references.push({
        phrase: mentionedComplement,
        itemIndices: rest,
        confidence: 1,
        via: "complement",
      });
    }
  }

  const { defaultRule, reason } = detectDefaultRule(instruction);

  return {
    references,
    defaultRule,
    defaultRuleReason: reason,
    promptBlock: renderPromptBlock(references, defaultRule, reason, items),
  };
}

/**
 * Renders resolved references for prompt injection. This is what turns
 * "search the list and decide which items are drinks" into "bind these
 * given indices to a person" — the form the bake-off measured at 7/9.
 *
 * Bug #1 found in the resolution-layer bake-off run: this used to
 * `return ""` whenever no ITEM reference resolved, which silently
 * dropped the DEFAULT RULE line too — even when detectDefaultRule() had
 * correctly worked out "exclude" from a marker like "just split ...
 * forget the rest". `exclude-unmentioned-items` failed for exactly this
 * reason: the right answer was computed and then never told to the
 * model. Fixed by decoupling the two.
 *
 * Bug #2, found in the very next run after fixing #1: always emitting
 * `DEFAULT RULE: equal` — including on cases where nothing needed
 * resolving and "equal" was only the code's own uninformative fallback
 * guess — added tokens that changed Llama-3.2-1B's greedy decode path to
 * a WRONG answer on `equal-baseline`, a case it previously got right.
 * Adding true-but-useless information is not free for a small model. The
 * line is now only rendered when it's actually informative: `exclude`
 * (proven necessary), or an `equal` that came from an explicit marker in
 * the text rather than the absence of one. See isInformativeDefaultRule.
 */
export function renderPromptBlock(
  references: ResolvedReference[],
  defaultRule: "equal" | "exclude",
  defaultRuleReason: string,
  items: ReceiptItem[],
): string {
  const lines = references
    .filter((r) => r.itemIndices.length > 0 && r.confidence >= AMBIGUITY_THRESHOLD)
    .map((r) => {
      const names = r.itemIndices.map((i) => items[i]?.name ?? `#${i}`).join(", ");
      const tag = r.isException ? " (stated as an exception)" : "";
      return `  "${r.phrase}" → items [${r.itemIndices.join(", ")}] (${names})${tag}`;
    });

  const header = "RESOLVED REFERENCES (already worked out for you — do not re-derive these):";
  const body = lines.length > 0 ? `${header}\n${lines.join("\n")}\n` : "";
  const ruleLine = isInformativeDefaultRule(defaultRule, defaultRuleReason)
    ? `DEFAULT RULE: ${defaultRule}`
    : "";

  return [body.trimEnd(), ruleLine].filter(Boolean).join("\n");
}
