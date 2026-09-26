import type { AssignmentPlan, ReceiptItem } from "@/types/domain";
import { isInformativeDefaultRule, resolveInstruction, type ResolvedReference } from "@/lib/retrieval/resolve";
import { normalise } from "@/lib/retrieval/lexicon";
import { findPersonMentions } from "@/lib/retrieval/people";
import { findAmbiguities, type Ambiguity } from "@/lib/ai/intentRouter";

// The no-LLM path: builds an AssignmentPlan from resolveInstruction()'s output
// plus a handful of clause rules. It serves guests / users without an API key,
// and is the last tier of the cascade when every cloud call fails.
//
// The contract that matters is "never guess": whatever it can't resolve with
// confidence comes back as a Chip (a question for the user), not as an
// assignment. Its accuracy is measured, not assumed — see fallback-parser.test.ts
// (development set = the 15 bake-off cases; held-out set = heldout-cases.ts).

export type Chip =
  | Ambiguity
  | { kind: "unresolved"; clause: string; reason: string }
  | { kind: "exception-unassigned"; phrase: string; options: { index: number; name: string }[] };

export interface FallbackResult {
  plan: AssignmentPlan;
  /** Questions for the user. Non-empty means the plan is incomplete or uncertain. */
  chips: Chip[];
  /**
   * True when at least one rule actually fired: an item was assigned, a tax payer named, or the
   * instruction carried an explicit "split equally" / "just … forget the rest" marker. False
   * means the parser just fell back to its blind default, so its plan says nothing about
   * whether another reading of the instruction is right — it must not be used to second-guess one.
   */
  understood: boolean;
}

// --- clause splitting ------------------------------------------------------

const SUBJECT_START = /^(?:we|everyone|everybody|he|she|they|i|the rest|everything else)\b/i;
const OWNERSHIP_VERB =
  /\b(?:pay|pays|paying|paid|had|has|have|having|owe|owes|owing|take|takes|took|get|gets|got|cover|covers|covering|order|orders|ordered|want|wants|ate|eat|eats|drank|split|splits|share|shares|treat|treats|treating)\b/i;

/**
 * Splits on `,` `;` dashes, and on " and " / " but " only where a NEW clause
 * starts — "between Pravin and Aisha" and "the mee goreng and the milo ais"
 * must stay whole, "Pravin had X and Aisha had Y" must not.
 */
export function splitClauses(instruction: string, people: string[]): string[] {
  const rough = instruction
    .split(/\s*[,;]\s*|\s+[—–-]\s+|\s*[—–]\s*|\.\s+/)
    .map((s) => s.trim())
    .filter(Boolean);

  const out: string[] = [];
  for (const piece of rough) {
    let rest = piece;
    for (;;) {
      const cut = findClauseBoundary(rest, people);
      if (!cut) break;
      out.push(rest.slice(0, cut.at).trim());
      rest = rest.slice(cut.after).trim();
    }
    out.push(rest);
  }
  return out
    .map((s) => s.replace(/^(?:and|but|so|then|also|plus)\s+/i, "").trim())
    .filter(Boolean);
}

function findClauseBoundary(text: string, people: string[]): { at: number; after: number } | null {
  const re = /\s+(and|but)\s+/gi;
  for (let m = re.exec(text); m; m = re.exec(text)) {
    const left = text.slice(0, m.index);
    const right = text.slice(m.index + m[0].length);
    if (!OWNERSHIP_VERB.test(left)) continue; // left has no verb → it's a list ("A and B")
    if (/\b(?:everyone|everybody|all|us)$/i.test(left.trim())) continue; // "everyone but Wei"
    const startsWithPerson = findPersonMentions(right.split(/\s+/).slice(0, 2).join(" "), people)
      .some((p) => p.index === 0);
    if (!(startsWithPerson || SUBJECT_START.test(right))) continue;
    if (!OWNERSHIP_VERB.test(right) && !/\b(?:is|are|s)\b/i.test(right)) continue;
    return { at: m.index, after: m.index + m[0].length };
  }
  return null;
}

// --- clause rules ----------------------------------------------------------

const NEGATION = /\b(?:not|no|never|didn t|doesn t|don t|won t|isn t|aren t|wasn t|without|skip|skipping)\b/;
const EVERYONE_EXCEPT = /\b(?:everyone|everybody|all of us|we all|all)\s+(?:except|but|apart from|other than|excluding|besides)\b/;
const EXCEPT_MARKER = /\b(?:except(?:\s+for)?|apart from|other than|excluding|but not)\b/;
const TREAT_ALL =
  /\b(?:treat|treats|treating|treated|foot|foots|footing)\b|\bput\s+(?:it\s+)?all\s+on\b|\b(?:on|all on)\s+(?:him|her)\b|\b(?:pay|pays|paying|paid|cover|covers|covering)\s+(?:for\s+)?(?:everything|it all|all of it|the whole (?:bill|thing|lot)|the entire bill|the lot)\b/;
// "Wifey pays the tax", "Aisha covers the service charge", "put the GST on him"
const TAX_MENTION = /\b(?:tax|taxes|gst|sst|vat|svc|service\s?charge|service\s?fee|service\s?tax)\b/;
const SPLITISH = /\b(?:split|splits|splitting|share|shares|sharing|divide|divides|equally|evenly|between|together|among|all|everyone|everybody|we|each)\b/;
const PRONOUN = /\b(?:he|she|him|her|his)\b/;
const SUBSET_OF_US = /\b(two|three|four|five|six|seven|eight|2|3|4|5|6|7|8)\s+of\s+(?:us|them)\b/;
const NUMBER_WORDS: Record<string, number> = {
  two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
};

const containsPhrase = (haystack: string, phrase: string) => ` ${haystack} `.includes(` ${normalise(phrase)} `);
const phraseIndex = (haystack: string, phrase: string) => ` ${haystack} `.indexOf(` ${normalise(phrase)} `);

export function parseInstruction(
  instruction: string,
  people: string[],
  items: ReceiptItem[],
): FallbackResult {
  const chips: Chip[] = [...findAmbiguities(instruction, people, items)];
  const resolution = resolveInstruction(instruction, items);
  const assign = new Map<number, string[]>();
  const allIdx = items.map((_, i) => i);
  const optionsFor = (indices: number[]) => indices.map((index) => ({ index, name: items[index].name }));

  const taxPayers = new Set<string>();
  let prevPerson: string | null = null;
  // Items named in the previous clause that nobody has been given yet: "…the
  // milkshake, which is just for Sarah" / "…the pizza, split between A and B".
  let dangling: { indices: number[]; isException: boolean; phrase: string }[] = [];
  const pendingExceptions: { phrase: string; indices: number[] }[] = [];

  for (const clause of splitClauses(instruction, people)) {
    const norm = normalise(clause);

    // --- who -------------------------------------------------------------
    let persons = findPersonMentions(clause, people).map((m) => m.person);
    const usedPronoun = persons.length === 0 && PRONOUN.test(norm) && prevPerson !== null;
    if (usedPronoun) persons = [prevPerson!];
    if (persons.length > 0) prevPerson = persons[persons.length - 1];

    // --- which items (clause-local view of the global resolution) ---------
    const excMarkerAt = EXCEPT_MARKER.exec(norm)?.index ?? -1;
    const positive: number[] = [];
    const exception: number[] = [];
    const clauseRefs: ResolvedReference[] = resolution.references.filter(
      (r) => r.itemIndices.length > 0 && r.confidence >= 0.5 && containsPhrase(norm, r.phrase),
    );
    for (const ref of clauseRefs) {
      const at = phraseIndex(norm, ref.phrase);
      const bucket = excMarkerAt >= 0 && at > excMarkerAt ? exception : positive;
      for (const i of ref.itemIndices) if (!bucket.includes(i)) bucket.push(i);
    }
    const onlyComplement = clauseRefs.length > 0 && clauseRefs.every((r) => r.via === "complement");
    const takeAll = (targets: string[], indices: number[]) => {
      for (const i of indices) assign.set(i, targets);
    };
    const remember = () => {
      dangling = [];
      if (!onlyComplement && positive.length > 0) dangling.push({ indices: positive, isException: false, phrase: clause });
      if (exception.length > 0) {
        dangling.push({ indices: exception, isException: true, phrase: clause });
        pendingExceptions.push({ phrase: clause, indices: exception });
      }
    };

    // 0. Who pays the tax/service charge. Tax isn't an item, so this never touches the item
    //    rules below — unless the same clause also names items ("Wifey pays the drinks and the tax").
    //    "…everything except the tax" is the opposite: the named person does NOT pay it.
    const taxAt = TAX_MENTION.exec(norm)?.index ?? -1;
    if (taxAt >= 0 && persons.length > 0 && !(excMarkerAt >= 0 && excMarkerAt < taxAt)) {
      for (const p of persons) taxPayers.add(p);
      if (positive.length === 0 && !TREAT_ALL.test(norm.replace(TAX_MENTION, ""))) {
        dangling = [];
        continue;
      }
    }

    // 1. "Everyone except Wei splits the cendol"
    const ee = EVERYONE_EXCEPT.exec(norm);
    if (ee) {
      const marker = ee.index + ee[0].length;
      const out = findPersonMentions(clause, people).filter((m) => m.index >= marker).map((m) => m.person);
      if (out.length > 0) {
        const group = people.filter((p) => !out.includes(p));
        if (group.length > 0) {
          // The except-marker here introduces a PERSON, so any item named after it
          // is the thing being split, not an item-exception.
          const named = [...positive, ...exception];
          takeAll(group, named.length > 0 ? named : allIdx);
          dangling = [];
          continue;
        }
      }
    }

    // 2. negation: "Wei didn't have the cendol", "she's not paying for the chicken"
    if (persons.length > 0 && positive.length > 0 && NEGATION.test(norm)) {
      const group = people.filter((p) => !persons.includes(p));
      if (group.length > 0) {
        takeAll(group, positive);
        dangling = [];
        continue;
      }
    }

    // 3. "Pravin treats everyone" / "put it all on him" / "Pravin pays for everything except X"
    if (persons.length > 0 && positive.length === 0 && TREAT_ALL.test(norm)) {
      takeAll(persons, allIdx.filter((i) => !exception.includes(i)));
      dangling = [];
      continue;
    }

    // 4. named people + named items: "Pravin pays for the drinks", "A and B share the pizza"
    if (persons.length > 0 && positive.length > 0) {
      takeAll(persons, positive);
      if (exception.length > 0) remember();
      else dangling = [];
      continue;
    }

    // 5. named people, no items in THIS clause: take whatever the last clause left dangling
    if (persons.length > 0 && positive.length === 0) {
      if (dangling.length > 0) {
        for (const d of dangling) takeAll(persons, d.indices);
        for (const d of dangling) {
          const k = pendingExceptions.findIndex((p) => p.indices === d.indices);
          if (k >= 0) pendingExceptions.splice(k, 1);
        }
        dangling = [];
      } else if (OWNERSHIP_VERB.test(clause)) {
        chips.push({
          kind: "unresolved",
          clause,
          reason: "I couldn't match what they had to an item on the receipt",
        });
      }
      continue;
    }

    // 6. items but nobody named, said as a group: "we all split the cake", "split the rest equally"
    if (positive.length > 0 && SPLITISH.test(norm)) {
      const subset = SUBSET_OF_US.exec(norm);
      if (subset && NUMBER_WORDS[subset[1]] !== people.length) {
        chips.push({ kind: "unresolved", clause, reason: `"${subset[0]}" — which of the ${people.length} people?` });
        continue;
      }
      takeAll(people, positive);
      dangling = [];
      continue;
    }

    // 7. items with no owner yet (or only exceptions) — wait for the next clause
    remember();
  }

  for (const p of pendingExceptions) {
    if (p.indices.every((i) => !assign.has(i))) {
      chips.push({ kind: "exception-unassigned", phrase: p.phrase, options: optionsFor(p.indices) });
    }
  }

  return {
    plan: {
      assignments: [...assign.entries()]
        .sort((a, b) => a[0] - b[0])
        .map(([itemIndex, who]) => ({ itemIndex, people: [...who] })),
      defaultRule: resolution.defaultRule,
      ...(taxPayers.size > 0 ? { taxPayers: [...taxPayers] } : {}),
      notes: "deterministic fallback parser",
    },
    chips,
    understood:
      assign.size > 0 ||
      taxPayers.size > 0 ||
      isInformativeDefaultRule(resolution.defaultRule, resolution.defaultRuleReason),
  };
}
