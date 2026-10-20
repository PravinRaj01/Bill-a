"use client";
import { useState } from "react";
import Link from "next/link";
import { AnimatePresence, m } from "motion/react";
import { ChevronDown, Check, Loader2, Send } from "lucide-react";

import { PersonAvatar } from "@/components/person-avatar";
import { formatMoney } from "@/lib/money";
import { shortDate, type PersonBalance } from "@/lib/split/settle";
import { cn } from "@/lib/utils";

const action =
  "flex min-h-11 items-center justify-center gap-2 rounded-xl border px-4 text-sm font-semibold transition-colors outline-none focus-visible:ring-2 focus-visible:ring-white/30 disabled:opacity-50";

// One person's balance with you across all tracked bills: who, how much (green = they owe you, coral = you
// owe them), how many bills, a Nudge (one message listing what they owe on each bill), Mark all paid, and
// the bills themselves on request.
export function BalanceCard({
  person,
  busy,
  claimBills,
  onNudge,
  onSettle,
}: {
  person: PersonBalance;
  busy: boolean;
  /** My bills with a friend's "I've paid" waiting, by bill id. */
  claimBills: Record<string, number>;
  onNudge: (p: PersonBalance) => void;
  onSettle: (p: PersonBalance) => void;
}) {
  const [open, setOpen] = useState(false);
  const theyOweYou = person.net > 0;
  const money = (c: number) => formatMoney(Math.abs(c), person.currency);
  const n = person.bills.length;
  // A friend can be both: some bills are mine (I can mark them paid), some are theirs (only they can).
  const hasOwn = person.bills.some((b) => !b.shared);
  const hasTheirs = person.bills.some((b) => b.shared);
  const ownCount = person.bills.filter((b) => !b.shared).length;
  const claimed = person.bills.find((b) => !b.shared && claimBills[b.id]);
  // Some bills they owe you, some you owe them: the card shows the net, and marking paid clears them all.
  const bothWays = person.bills.some((b) => b.amount > 0) && person.bills.some((b) => b.amount < 0);

  return (
    <li data-testid="balance-card" data-person={person.name} className="space-y-4 rounded-3xl glass p-5">
      <div className="flex items-center gap-3">
        <PersonAvatar name={person.name} className="size-11" />
        <div className="min-w-0 flex-1">
          <h3 className="truncate text-lg font-bold text-foreground">{person.name}</h3>
          <p className="text-sm text-muted-foreground">
            {person.shared ? "shared with you · " : ""}across {n} {n === 1 ? "bill" : "bills"}
          </p>
        </div>
        <div className="shrink-0 text-right">
          <p data-testid="balance-amount" className={cn("text-xl font-bold tabular-nums", theyOweYou ? "text-success" : "text-coral")}>
            {money(person.net)}
          </p>
          <p className="text-xs font-semibold text-muted-foreground">{theyOweYou ? "owes you" : "you owe"}</p>
        </div>
      </div>

      {claimed && (
        <Link href={`/dashboard/history/${claimed.id}?from=balances`} transitionTypes={["nav-forward"]} data-testid="balance-claim" className="-mt-1 flex min-h-11 items-center justify-between gap-2 rounded-2xl border border-success/30 bg-success/10 px-4 text-sm font-semibold text-success">
          <span>{person.name} says they paid you</span>
          <span>Check</span>
        </Link>
      )}

      {bothWays && (
        <p data-testid="both-ways" className="-mt-1 px-1 text-xs text-muted-foreground">
          Bills go both ways, so this is the net. Marking paid settles {hasTheirs ? `your ${ownCount}` : `all ${n}`} of them.
        </p>
      )}

      {hasTheirs && (
        <p data-testid="shared-note" className="-mt-1 px-1 text-xs text-muted-foreground">
          {hasOwn ? `Some of these are ${person.name}'s bills,` : `These are ${person.name}'s bills,`} so only they can mark those paid. Open one to say you&apos;ve paid.
        </p>
      )}

      <div className="flex flex-wrap gap-2">
        {theyOweYou && hasOwn && (
          <button
            type="button"
            data-testid="balance-nudge"
            onClick={() => onNudge(person)}
            className={cn(action, "flex-1 border-white/10 text-foreground hover:bg-white/5")}
          >
            <Send className="size-4" /> Nudge {person.name}
          </button>
        )}
        {hasOwn && (
        <button
          type="button"
          data-testid="balance-settle"
          disabled={busy}
          onClick={() => onSettle(person)}
          className={cn(action, "flex-1 border-success/40 text-success hover:bg-success/10")}
        >
          {busy ? <Loader2 className="size-4 animate-spin" /> : <Check className="size-4" strokeWidth={3} />}
          {theyOweYou ? "Mark all paid" : "I've paid them"}
        </button>
        )}
      </div>

      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        className="flex min-h-11 w-full items-center justify-between px-1 text-sm font-semibold text-muted-foreground transition-colors hover:text-foreground"
      >
        {open ? "Hide bills" : `Show ${n === 1 ? "the bill" : `${n} bills`}`}
        <ChevronDown className={cn("size-4 transition-transform", open && "rotate-180")} />
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <m.ul
            key="bills"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.22 }}
            className="-mt-2 space-y-1 overflow-hidden"
          >
            {person.bills.map((b) => (
              <li key={b.id}>
                <Link
                  href={b.shared ? `/dashboard/shared/${b.id}` : `/dashboard/history/${b.id}?from=balances`}
                  transitionTypes={["nav-forward"]}
                  className="flex min-h-12 items-center justify-between gap-3 rounded-xl bg-white/[0.07] px-3 py-2 transition-colors hover:bg-white/5"
                >
                  <span className="min-w-0">
                    <span className="block truncate text-sm font-semibold text-foreground">{b.title}</span>
                    <span className="block text-xs text-muted-foreground">
                      {shortDate(b.createdAt)}
                      {b.dueDate ? ` · due ${shortDate(b.dueDate)}` : ""}
                    </span>
                  </span>
                  <span className={cn("shrink-0 text-sm font-bold tabular-nums", b.amount > 0 ? "text-success" : "text-coral")}>
                    {b.amount > 0 ? "+" : "−"}
                    {money(b.amount)}
                  </span>
                </Link>
              </li>
            ))}
          </m.ul>
        )}
      </AnimatePresence>
    </li>
  );
}
