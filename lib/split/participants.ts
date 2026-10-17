import type { BillData, Participant } from "@/types/domain";

// Who is in a split, and which one is the user. Before batch 5 the app guessed "me" by first name or asked
// "Which one is you?"; now every new split starts with a Me chip, and the bill records it. These rules are
// shared by the server validation, the split session and the screens so they can't drift apart.

const norm = (s: string) => s.trim().toLowerCase();

/** The same person, ignoring case and surrounding spaces (as balances already do). */
export const samePerson = (a: string, b: string): boolean => norm(a) === norm(b);

/** What the user is called inside a split: the first word of their account name, else "Me". */
export function meNameFor(accountName?: string | null): string {
  const first = accountName?.trim().split(/\s+/)[0];
  return first && first.length <= 60 ? first : "Me";
}

/** The people of a split as participants, marking the one that is the user (null = split for others). */
export function buildParticipants(people: string[], meName: string | null): Participant[] {
  let marked = false;
  return people.map((name) => {
    if (meName && !marked && samePerson(name, meName)) {
      marked = true;
      return { name, me: true as const };
    }
    return { name };
  });
}

type WithParticipants = Pick<BillData, "participants" | "settlement">;

/** The user's name in this bill: from the participants when it has them, else the old `settlement.me` guess. */
export const billMe = (data: WithParticipants): string | undefined => (data.participants ? data.participants.find((p) => p.me)?.name : data.settlement?.me);

/** Split for others: the bill knows its participants and the user is not one of them. */
export const isHosted = (data: Pick<BillData, "participants">): boolean => !!data.participants && !data.participants.some((p) => p.me);

/** Why a typed name can't be added, or null when it can. */
export type NameProblem = "empty" | "duplicate" | "you" | "arrow";
export function nameProblem(raw: string, existing: string[], meName: string | null): NameProblem | null {
  const name = raw.trim();
  if (!name) return "empty";
  // "Alex->Sam" is how a paid transfer is keyed; a name containing it would corrupt those keys.
  if (name.includes("->")) return "arrow";
  if (existing.some((p) => samePerson(p, name))) return meName && samePerson(name, meName) ? "you" : "duplicate";
  return null;
}

/**
 * Checks the participants of a bill, or null when fine. Applies only to bills that HAVE participants, so
 * older saved bills (and anything still in someone's outbox) keep syncing.
 */
export function participantProblem(data: Pick<BillData, "participants" | "people" | "settlement">): string | null {
  const parts = data.participants;
  if (!parts) return null;
  if (parts.length !== data.people.length || parts.some((p, i) => p.name !== data.people[i])) return "participants must match people";
  const seen = new Set<string>();
  for (const p of parts) {
    const key = norm(p.name);
    if (!key) return "empty name";
    if (p.name.includes("->")) return "a name can't contain ->";
    if (seen.has(key)) return "duplicate name";
    seen.add(key);
  }
  const mine = parts.filter((p) => p.me);
  if (mine.length > 1) return "more than one me";
  if (mine[0]?.userId) return "me can't link another account";
  // settlement.me stays written for older readers, so it must agree with the participants.
  const settledMe = data.settlement?.me;
  if (mine[0] ? settledMe !== undefined && settledMe !== mine[0].name : settledMe !== undefined) return "settlement.me disagrees with participants";
  return null;
}

/**
 * In an instruction ("I didn't have the rice", "split it between me and Riley"), "I", "me" and "myself"
 * mean the user: swap them for the Me name so the reader finds a person. Contractions (I'm, I'd) are left
 * alone, and nothing changes when the user isn't in the split.
 */
export function resolveSelfRefs(text: string, meName: string | null, people: string[]): string {
  if (!meName || meName === "Me" || !people.some((p) => samePerson(p, meName))) return text;
  return text.replace(/\b(?:i|me|myself)\b(?!['’])/gi, meName);
}
