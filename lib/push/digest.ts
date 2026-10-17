import { formatMoney } from "@/lib/money";
import type { PersonBalance } from "@/lib/split/settle";

// The once-a-day summary. Built from what the app already knows, and only when there is something to say:
// an empty "nothing today" push every morning would get the feature switched off.

export const DIGEST_HOUR = 9; // local time: the digest goes out from this hour on, once per day

export interface DigestInput {
  people: Pick<PersonBalance, "currency" | "net">[];
  unread: number;
  claimsToConfirm: number;
}

const totals = (people: DigestInput["people"], sign: 1 | -1) => {
  const by = new Map<string, number>();
  for (const p of people) if (p.net * sign > 0) by.set(p.currency, (by.get(p.currency) ?? 0) + Math.abs(p.net));
  return [...by.entries()].map(([currency, cents]) => formatMoney(cents, currency)).join(" + ");
};

export function buildDigest(i: DigestInput): { title: string; body: string; url: string } | null {
  const owed = totals(i.people, 1);
  const owe = totals(i.people, -1);
  const parts = [
    owed && `Friends owe you ${owed}`,
    owe && `You owe ${owe}`,
    i.claimsToConfirm > 0 && `${i.claimsToConfirm} payment${i.claimsToConfirm === 1 ? "" : "s"} to confirm`,
    i.unread > 0 && `${i.unread} unread`,
  ].filter(Boolean) as string[];
  if (parts.length === 0) return null;
  const url = i.claimsToConfirm > 0 || i.unread > 0 ? "/dashboard/notifications" : "/dashboard/balances";
  return { title: "Bill.a daily summary", body: parts.join(" · "), url };
}

/** The user's own calendar day and hour right now, from their IANA zone (falls back to UTC for a bad zone). */
export function localDayHour(tz: string, now = new Date()): { day: string; hour: number } {
  let parts: Intl.DateTimeFormatPart[];
  try {
    parts = new Intl.DateTimeFormat("en-CA", { timeZone: tz, year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", hourCycle: "h23" }).formatToParts(now);
  } catch {
    return localDayHour("UTC", now);
  }
  const get = (t: string) => parts.find((p) => p.type === t)!.value;
  return { day: `${get("year")}-${get("month")}-${get("day")}`, hour: Number(get("hour")) };
}
