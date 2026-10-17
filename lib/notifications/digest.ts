import { formatMoney } from "@/lib/money";
import type { PersonBalance } from "@/lib/split/settle";

// The once-a-day summary: who still owes you, who you owe, and what is due today. Built from the user's balances,
// so it is only ever about the user's own money. Nothing to say means no digest at all.

const FALLBACK_TZ = "Asia/Kuala_Lumpur";

/** Today's date (YYYY-MM-DD) where this person lives, so "due today" means their today. */
export function dateInZone(now: Date, tz: string): string {
  for (const zone of [tz, FALLBACK_TZ]) {
    try {
      return new Intl.DateTimeFormat("en-CA", { timeZone: zone, year: "numeric", month: "2-digit", day: "2-digit" }).format(now);
    } catch {
      /* an unknown zone: try the default */
    }
  }
  return now.toISOString().slice(0, 10);
}

interface Side {
  people: PersonBalance[];
  total: number;
  dueToday: number;
}

const side = (people: PersonBalance[], sign: 1 | -1, today: string): Side => ({
  people,
  total: people.reduce((n, p) => n + Math.abs(p.net), 0),
  dueToday: people.reduce((n, p) => n + p.bills.filter((b) => Math.sign(b.amount) === sign && b.dueDate && b.dueDate <= today).reduce((m, b) => m + Math.abs(b.amount), 0), 0),
});

const due = (s: Side, currency: string) => (s.dueToday === 0 ? "" : s.dueToday >= s.total ? ", due today" : `, ${formatMoney(s.dueToday, currency)} due today`);

/** One sentence per currency (never netted across), joined with " · ". Null when nobody owes anything. */
export function digestSummary(balances: PersonBalance[], today: string): string | null {
  const currencies = [...new Set(balances.map((b) => b.currency))];
  const parts: string[] = [];
  for (const currency of currencies) {
    const mine = balances.filter((b) => b.currency === currency);
    const owed = side(mine.filter((b) => b.net > 0), 1, today);
    const owe = side(mine.filter((b) => b.net < 0), -1, today);
    if (owed.people.length) {
      const who = owed.people.length === 1 ? `${owed.people[0].name} owes` : `${owed.people.length} people owe`;
      parts.push(`${who} you ${formatMoney(owed.total, currency)}${due(owed, currency)}`);
    }
    if (owe.people.length) {
      const who = owe.people.length === 1 ? owe.people[0].name : `${owe.people.length} people`;
      parts.push(`You owe ${who} ${formatMoney(owe.total, currency)}${due(owe, currency)}`);
    }
  }
  return parts.length ? parts.join(" · ") : null;
}
