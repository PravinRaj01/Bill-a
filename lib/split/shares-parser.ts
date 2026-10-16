import type { Cents, ReceiptItem } from "@/types/domain";
import { resolveInstruction } from "@/lib/retrieval/resolve";
import { apportion } from "./apportion";
import type { Chip } from "./fallback-parser";

// Reads whole-bill shares out of an instruction, deterministically and on the device:
//   "Pravin 70%, Lakx 30%"            "pravin pays 70% lakx pay 30%"
//   "70/30 Pravin and Lakx"           "split 2:1 between A and B"
//   "Lakx pays RM30, rest equally"    "Pravin 60%, Lakx gets the rest"
// A model is never involved: this is arithmetic on numbers the user typed. What it can't
// read with confidence comes back as a question (a chip with one-tap fixes), never a guess.
// Anything that is really about ITEMS ("Pravin pays 50% of the drinks", "the tax") is left
// alone (kind "none") for the item-based pipeline.

export type ShareParse =
  | { kind: "none" }
  | { kind: "ask"; chips: Chip[] }
  | {
      kind: "shares";
      /** Aligned with `people`. Sums to `total` exactly. */
      amounts: Cents[];
      /** Aligned with `people`: how each person's share came about ("" = not part of this split). */
      labels: string[];
      /** Plain-words lines for the reasoning log. */
      notes: string[];
    };

interface Term {
  kind: "pct" | "money";
  /** Percent as written, or cents. */
  value: number;
  index: number;
  end: number;
}
interface RatioTerm {
  numbers: number[];
  index: number;
  end: number;
}
interface Mention {
  person: string;
  index: number;
  end: number;
}

const PCT = /(\d+(?:\.\d+)?)\s*(?:%|per\s?cent\b|percent\b|pct\b)/giu;
const MONEY = /(?:\b(?:rm|myr)|\$)\s*(\d+(?:[.,]\d{1,2})?)|(\d+(?:\.\d{1,2})?)\s*(?:rm|myr|ringgit|dollars?|bucks)\b/giu;
const RATIO = /(?<![\d.])\d+(?:\.\d+)?(?:\s*[:/]\s*\d+(?:\.\d+)?)+(?![\d%])/gu;

// Talking about tax/service/discounts is the item pipeline's job.
const TAX_WORDS = /\b(?:tax|taxes|gst|sst|vat|svc|service|tip|discount|off)\b/i;
const REST_WORDS = /\b(?:rest|remaining|remainder|others?|everyone else|everybody else|balance|left\s?over)\b/i;

const escapeRe = (t: string) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const cents = (n: number) => Math.round(n * 100);
const money = (c: Cents) => `RM${(c / 100).toFixed(2)}`;
const pctText = (p: number) => `${+p.toFixed(2)}%`;

function findTerms(text: string) {
  const terms: Term[] = [];
  for (const m of text.matchAll(PCT)) {
    terms.push({ kind: "pct", value: parseFloat(m[1]), index: m.index, end: m.index + m[0].length });
  }
  for (const m of text.matchAll(MONEY)) {
    const raw = (m[1] ?? m[2]).replace(",", ".");
    terms.push({ kind: "money", value: cents(parseFloat(raw)), index: m.index, end: m.index + m[0].length });
  }
  const ratios: RatioTerm[] = [];
  for (const m of text.matchAll(RATIO)) {
    ratios.push({ numbers: m[0].split(/[:/]/).map((n) => parseFloat(n)), index: m.index, end: m.index + m[0].length });
  }
  return { terms, ratios };
}

/** Where each person is named in the raw text (full name, or a first name that is unique in the group). */
function locatePeople(text: string, people: string[]): Mention[] {
  const firstOf = (p: string) => p.trim().split(/\s+/)[0];
  const firsts = people.map((p) => firstOf(p).toLowerCase());
  const found: (Mention & { len: number })[] = [];
  people.forEach((person, i) => {
    const names = new Set([person.trim()]);
    if (firsts.filter((f) => f === firsts[i]).length === 1) names.add(firstOf(person));
    for (const name of names) {
      if (!name) continue;
      const re = new RegExp(`(?<![\\p{L}\\p{N}])${escapeRe(name)}(?![\\p{L}\\p{N}])`, "giu");
      for (const m of text.matchAll(re)) found.push({ person, index: m.index, end: m.index + m[0].length, len: m[0].length });
    }
  });
  // "Wei Jie" also matches "Wei": keep the longer, drop what it overlaps.
  found.sort((a, b) => a.index - b.index || b.len - a.len);
  const out: Mention[] = [];
  for (const f of found) if (!out.some((o) => f.index < o.end && f.end > o.index)) out.push(f);
  return out;
}

const ask = (message: string, fixes: { label: string; append: string }[] = []): ShareParse => ({
  kind: "ask",
  chips: [{ kind: "shares", message, fixes }],
});

export function parseShareInstruction(
  text: string,
  people: string[],
  total: Cents,
  items: ReceiptItem[] = [],
): ShareParse {
  const { terms, ratios } = findTerms(text);
  if (terms.length === 0 && ratios.length === 0) return { kind: "none" };
  if (TAX_WORDS.test(text)) return { kind: "none" };
  // A named item means this is about ITEMS ("Pravin pays 50% of the drinks"), not the whole bill.
  // ("the rest" resolves as a complement of other items; it isn't an item.)
  if (items.length > 0 && resolveInstruction(text, items).references.some((r) => r.via !== "complement" && r.itemIndices.length > 0 && r.confidence >= 0.5)) {
    return { kind: "none" };
  }

  const mentions = locatePeople(text, people);
  if (mentions.length === 0) return { kind: "none" }; // numbers with nobody to give them to
  if (total <= 0) return ask("There's nothing to split yet — the total is 0.");

  const restWord = REST_WORDS.test(text);

  // ---- ratio: "70/30 Pravin and Lakx", "split 2:1 between A and B" -------------------
  if (ratios.length > 0) {
    if (terms.length > 0 || ratios.length > 1) return ask("Use one way of giving shares: percentages, a ratio, or amounts — not a mix.");
    const r = ratios[0];
    const order = [...new Set(mentions.map((m) => m.person))];
    if (r.numbers.length !== order.length) {
      return ask(`I see ${r.numbers.length} numbers (${r.numbers.join(":")}) but ${order.length} ${order.length === 1 ? "person" : "people"} named. Say it like “Pravin 70%, Lakx 30%”.`);
    }
    if (r.numbers.some((n) => !(n > 0))) return ask("Every number in a ratio needs to be above 0.");
    const weights = people.map((p) => {
      const k = order.indexOf(p);
      return k >= 0 ? r.numbers[k] : 0;
    });
    const sum = r.numbers.reduce((a, b) => a + b, 0);
    const amounts = apportion(total, weights);
    return {
      kind: "shares",
      amounts,
      labels: people.map((p) => {
        const k = order.indexOf(p);
        return k >= 0 ? `${r.numbers[k]} of ${+sum.toFixed(2)} parts` : "";
      }),
      notes: [`Ratio ${r.numbers.join(":")} between ${order.join(", ")} of ${money(total)}.`, ...notPartNote(people, order)],
    };
  }

  // ---- pair each number with the person it belongs to ---------------------------------
  type Tok = { t: "P"; m: Mention } | { t: "N"; term: Term };
  const seq: Tok[] = [
    ...mentions.map((m): Tok => ({ t: "P", m })),
    ...terms.map((term): Tok => ({ t: "N", term })),
  ].sort((a, b) => (a.t === "P" ? a.m.index : a.term.index) - (b.t === "P" ? b.m.index : b.term.index));

  const pairs: { person: string; term: Term }[] = [];
  const unpairedPeople: string[] = [];
  const unpairedTerms: Term[] = [];
  let pendingPerson: Mention | null = null;
  let pendingTerm: Term | null = null;
  for (const tok of seq) {
    if (tok.t === "P") {
      if (pendingTerm) {
        pairs.push({ person: tok.m.person, term: pendingTerm });
        pendingTerm = null;
      } else {
        if (pendingPerson) unpairedPeople.push(pendingPerson.person);
        pendingPerson = tok.m;
      }
    } else if (pendingPerson) {
      pairs.push({ person: pendingPerson.person, term: tok.term });
      pendingPerson = null;
    } else {
      if (pendingTerm) unpairedTerms.push(pendingTerm);
      pendingTerm = tok.term;
    }
  }
  if (pendingPerson) unpairedPeople.push(pendingPerson.person);
  if (pendingTerm) unpairedTerms.push(pendingTerm);

  if (pairs.length === 0) return { kind: "none" };
  if (unpairedTerms.length > 0) {
    const t = unpairedTerms[0];
    return ask(`I couldn't tell who the ${t.kind === "pct" ? pctText(t.value) : money(t.value)} is for. Say it like “Pravin 70%, Lakx 30%”.`);
  }
  const named = pairs.map((p) => p.person);
  if (new Set(named).size !== named.length) return ask("Someone has two shares in that instruction. Give each person one.");
  if (new Set(pairs.map((p) => p.term.kind)).size > 1) return ask("Use one way of giving shares: percentages or amounts — not a mix.");

  const given = new Map(pairs.map((p) => [p.person, p.term.value]));
  const isPct = pairs[0].term.kind === "pct";
  const restPeopleNamed = [...new Set(unpairedPeople)].filter((p) => !given.has(p));
  const others = people.filter((p) => !given.has(p));
  // Who gets what is left: people named without a number ("Lakx gets the rest"), otherwise everyone not given a share.
  const takers = restWord ? (restPeopleNamed.length > 0 ? restPeopleNamed : others) : [];

  if (restPeopleNamed.length > 0 && !restWord) {
    return ask(
      `${restPeopleNamed.join(" and ")} ${restPeopleNamed.length > 1 ? "have" : "has"} no share yet.`,
      restPeopleNamed.map((p) => ({ label: `${p} gets the rest`, append: `${p} gets the rest` })),
    );
  }

  // ---- percentages ---------------------------------------------------------------------
  if (isPct) {
    const sum = [...given.values()].reduce((a, b) => a + b, 0);
    if ([...given.values()].some((v) => v > 100)) return ask("A share can't be more than 100%.");
    const eps = 1e-6;
    let weights: number[];
    const labels: string[] = [];
    const notes: string[] = [`Shares you gave: ${pairs.map((p) => `${p.person} ${pctText(p.term.value)}`).join(", ")}.`];

    if (takers.length > 0) {
      const remaining = 100 - sum;
      if (remaining <= eps) return ask(`${pairs.map((p) => pctText(p.term.value)).join(" + ")} already makes ${pctText(sum)}, so nothing is left for ${takers.join(", ")}.`);
      const each = remaining / takers.length;
      weights = people.map((p) => (given.has(p) ? given.get(p)! : takers.includes(p) ? each : 0));
      people.forEach((p) => labels.push(given.has(p) ? `${pctText(given.get(p)!)} of the bill` : takers.includes(p) ? `${takers.length > 1 ? "equal share of" : ""} the remaining ${pctText(remaining)}`.trim() : ""));
      notes.push(`The remaining ${pctText(remaining)} is shared ${takers.length > 1 ? "equally " : ""}by ${takers.join(", ")}.`);
    } else {
      if (Math.abs(sum - 100) > eps) {
        if (sum > 100) return ask(`Those add up to ${pctText(sum)}, which is more than 100%.`);
        const left = 100 - sum;
        return ask(
          `Those add up to ${pctText(sum)}; ${pctText(left)} is left over.`,
          others.length > 0 ? [{ label: `Share the other ${pctText(left)} equally (${others.join(", ")})`, append: "the rest equally" }] : [],
        );
      }
      weights = people.map((p) => given.get(p) ?? 0);
      people.forEach((p) => labels.push(given.has(p) ? `${pctText(given.get(p)!)} of the bill` : ""));
    }
    notes.push(...notPartNote(people, [...given.keys(), ...takers]));
    return { kind: "shares", amounts: apportion(total, weights), labels, notes };
  }

  // ---- fixed amounts -------------------------------------------------------------------
  const fixed = [...given.values()].reduce((a, b) => a + b, 0);
  const left = total - fixed;
  if (left < 0) return ask(`Those add up to ${money(fixed)}, which is more than the bill (${money(total)}).`);
  const notes: string[] = [`Amounts you gave: ${pairs.map((p) => `${p.person} ${money(p.term.value)}`).join(", ")}.`];
  const amounts = people.map((p) => given.get(p) ?? 0);
  const labels: string[] = people.map((p) => (given.has(p) ? "amount you gave" : ""));

  if (takers.length > 0) {
    const split = apportion(left, takers.map(() => 1));
    takers.forEach((p, k) => {
      const i = people.indexOf(p);
      amounts[i] = split[k];
      labels[i] = `${takers.length > 1 ? "equal share of " : ""}the remaining ${money(left)}`.trim();
    });
    notes.push(`The remaining ${money(left)} is shared ${takers.length > 1 ? "equally " : ""}by ${takers.join(", ")}.`);
  } else if (left !== 0) {
    return ask(
      `${money(fixed)} is covered; ${money(left)} is left over. Who pays the rest?`,
      others.length > 0 ? [{ label: `Share the rest equally (${others.join(", ")})`, append: "the rest equally" }] : [],
    );
  }
  notes.push(...notPartNote(people, [...given.keys(), ...takers]));
  return { kind: "shares", amounts, labels, notes };
}

function notPartNote(people: string[], inSplit: string[]): string[] {
  const out = people.filter((p) => !inSplit.includes(p));
  return out.length > 0 ? [`${out.join(", ")} ${out.length > 1 ? "aren't" : "isn't"} part of this split.`] : [];
}
