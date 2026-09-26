import { CATEGORIES, normalise } from "./lexicon";

// Person-name matching shared by the intent router and the fallback parser.
// Exact matches are trusted; near misses are only ever SUGGESTED (a chip asks
// "did you mean Pravin?") — a typo is never silently resolved to a person,
// because assigning money to the wrong person is the one error that matters.

export function levenshtein(a: string, b: string): number {
  if (a === b) return 0;
  if (!a.length) return b.length;
  if (!b.length) return a.length;
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) {
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    }
    prev = cur;
  }
  return prev[b.length];
}

/** Words that appear in instructions and must never be mistaken for a name. */
const NOT_NAMES = new Set(
  (
    "everyone everybody everything anyone anything all we us our i me my mine you your he she him her his hers they them their " +
    "the a an and or but so if then also just only even each both rest others other else same equally evenly " +
    "split share shares sharing pay pays paying paid owe owes owing had has have having take takes taking took get gets got " +
    "cover covers covering treat treats treating put add invite include except apart from than not no never dont doesnt didnt " +
    "wont isnt arent did do does is are was were be been for on of to with without between among into per about ignore forget leave out " +
    "please can could would should will shall may might must who whom whose which that this these those it its there here " +
    "food drinks drink extra order bill table today tonight lunch dinner breakfast " +
    "tax taxes gst sst vat svc charge fee service tip"
  ).split(" "),
);

const OWNERSHIP_VERBS = new Set(
  "pay pays paying paid owe owes owing had has took takes ordered ate drank gets got covers wants".split(" "),
);

/** Every word that appears in an item's name or in the category lexicon. */
export function knownWords(itemNames: string[]): Set<string> {
  const words = new Set<string>();
  for (const name of itemNames) for (const t of normalise(name).split(" ")) words.add(t);
  for (const cat of CATEGORIES) {
    for (const kw of [...cat.itemKeywords, ...cat.queryPhrases]) {
      for (const t of normalise(kw).split(" ")) words.add(t);
    }
  }
  return words;
}

const firstToken = (name: string) => normalise(name).split(" ")[0] ?? "";

export interface PersonMention {
  person: string;
  /** Character offset into the normalised text. */
  index: number;
}

/**
 * People named in `text` (exact, case-insensitive, whole-word; a possessive or
 * contraction like "Pravin's" counts), in order of appearance. A person matches
 * on their full name, or on their first name when that first name is unique in
 * the group.
 */
export function findPersonMentions(text: string, people: string[]): PersonMention[] {
  const hay = ` ${normalise(text)} `;
  const firsts = people.map(firstToken);
  const out: PersonMention[] = [];
  people.forEach((person, i) => {
    const full = normalise(person);
    const candidates = new Set([full]);
    if (firsts.filter((f) => f === firsts[i]).length === 1) candidates.add(firsts[i]);
    let best = -1;
    for (const cand of candidates) {
      if (!cand) continue;
      const at = hay.indexOf(` ${cand} `);
      if (at >= 0 && (best < 0 || at < best)) best = at;
    }
    // normalise() turns "Pravin's" into "pravin s", so the name is followed by a
    // space and matches above; nothing extra is needed for possessives.
    if (best >= 0) out.push({ person, index: best });
  });
  return out.sort((a, b) => a.index - b.index);
}

export interface NameProblem {
  /** The word as typed. */
  token: string;
  /** Someone in the group it is probably a typo of, if close enough. */
  suggestion?: string;
}

/**
 * Words in `instruction` that look like a person but aren't in the group:
 *  - a token 1–2 edits away from a real name (a typo) → with a suggestion;
 *  - a capitalised word mid-sentence that is no known word → unknown person.
 */
export function findNameProblems(
  instruction: string,
  people: string[],
  itemNames: string[],
): NameProblem[] {
  const known = knownWords(itemNames);
  const exact = new Set(people.flatMap((p) => [normalise(p), firstToken(p), ...normalise(p).split(" ")]));
  const problems: NameProblem[] = [];
  const seen = new Set<string>();

  const rawTokens = instruction.split(/[^A-Za-z0-9']+/).filter(Boolean);
  rawTokens.forEach((raw, idx) => {
    // "Sarah's" -> "sarah"; "didn't" -> "did"
    const token = normalise(raw.replace(/n't$/i, "").replace(/'\w+$/, "")).replace(/\s.*/, "");
    if (token.length < 3 || seen.has(token)) return;
    if (exact.has(token) || NOT_NAMES.has(token) || known.has(token)) return;

    let suggestion: string | undefined;
    if (token.length >= 4) {
      const limit = token.length >= 7 ? 2 : 1;
      let bestDist = Infinity;
      for (const p of people) {
        for (const part of new Set([firstToken(p), normalise(p)])) {
          const d = levenshtein(token, part);
          if (d > 0 && d <= limit && d < bestDist) {
            bestDist = d;
            suggestion = p;
          }
        }
      }
    }

    const capitalisedMidSentence = idx > 0 && /^[A-Z]/.test(raw);
    // A word in the subject position of an ownership verb ("Wei owes…", "wei had…")
    // is a person whatever its capitalisation or position.
    const nextToken = normalise((rawTokens[idx + 1] ?? "").replace(/'\w+$/, "")).split(" ")[0];
    const isSubject = OWNERSHIP_VERBS.has(nextToken);
    if (suggestion || capitalisedMidSentence || isSubject) {
      seen.add(token);
      problems.push({ token: raw.replace(/'s$/i, ""), suggestion });
    }
  });
  return problems;
}
