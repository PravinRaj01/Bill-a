import { samePerson } from "@/lib/split/participants";
import type { BillData } from "@/types/domain";

// When a friend changes their name, the bills that name them follow. A bill keeps names as text in several
// places (shares, payers, paid marks, ...), so renaming one person means rewriting all of them together.

/**
 * What a name someone else wrote for this person becomes after they rename: their old full name becomes the new
 * full name, their old first name the new first name. Anything else (a nickname, their @username) was chosen by
 * the writer and is left alone (null).
 */
export function renamedForm(written: string, oldName: string | null, newName: string | null): string | null {
  const from = (oldName ?? "").trim().replace(/\s+/g, " ");
  const to = (newName ?? "").trim().replace(/\s+/g, " ");
  if (!from || !to || samePerson(from, to)) return null;
  if (samePerson(written, from)) return to;
  const oldFirst = from.split(" ")[0];
  const newFirst = to.split(" ")[0];
  if (samePerson(written, oldFirst) && !samePerson(oldFirst, newFirst)) return newFirst;
  return null;
}

/**
 * The bill with person `from` renamed to `to` everywhere it names them. Null when there is nothing to do, or
 * when `to` is already someone else on the bill (merging two people would change who owes what).
 */
export function renameInBill(data: BillData, from: string, to: string): BillData | null {
  if (samePerson(from, to) || !to.trim()) return null;
  const names = [...(data.split ?? []).map((s) => s.name), ...(data.people ?? [])];
  if (!names.some((n) => samePerson(n, from))) return null;
  if (names.some((n) => samePerson(n, to))) return null;

  const r = (n: string) => (samePerson(n, from) ? to : n);
  const out: BillData = {
    ...data,
    split: (data.split ?? []).map((s) => ({ ...s, name: r(s.name) })),
    people: (data.people ?? []).map(r),
  };
  if (data.exactSplit) out.exactSplit = data.exactSplit.map((s) => ({ ...s, name: r(s.name) }));
  if (data.cashRounding) out.cashRounding = { ...data.cashRounding, absorber: r(data.cashRounding.absorber) };
  if (data.participants) out.participants = data.participants.map((p) => ({ ...p, name: r(p.name) }));
  if (data.settlement) {
    const s = data.settlement;
    out.settlement = {
      ...s,
      ...(s.me !== undefined ? { me: r(s.me) } : {}),
      paidBy: (s.paidBy ?? []).map((p) => ({ ...p, name: r(p.name) })),
      paid: Object.fromEntries(Object.entries(s.paid ?? {}).map(([k, v]) => [renameTransferKey(k, from, to), v])),
    };
  }
  return out;
}

/** "From->To" with the person renamed on either side. */
export function renameTransferKey(key: string, from: string, to: string): string {
  const i = key.indexOf("->");
  if (i < 0) return key;
  const a = key.slice(0, i);
  const b = key.slice(i + 2);
  return `${samePerson(a, from) ? to : a}->${samePerson(b, from) ? to : b}`;
}
