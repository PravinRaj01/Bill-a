"use client";
import { useState, type ReactNode } from "react";
import { AnimatePresence, m } from "motion/react";
import { ChevronDown } from "lucide-react";

import { AvatarStack, PersonAvatar } from "@/components/person-avatar";
import { Money } from "@/components/ui/money";
import { parseItems } from "@/lib/split/items";
import { shortDate } from "@/lib/split/settle";
import { cn } from "@/lib/utils";

export interface TicketRow {
  name: string;
  /** Cents. */
  amount: number;
  /** The engine's item string for this person (see lib/split/items.ts). */
  items: string;
  /** Extra lines under the name: who paid, rounding, and so on. */
  note?: ReactNode;
}

// The bill as a paper receipt: a header with the title, date, total and who is splitting, a perforated
// tear line, then one row per person that opens to show what they had. The same ticket is used on the
// Summary (live session) and the History detail (saved bill); `children` are the cards that follow
// (settle up, adjusting, sharing) inside the same sheet.
export function ReceiptTicket({
  title,
  date,
  symbol,
  rows,
  children,
}: {
  title: string;
  date: string | number | Date;
  symbol: string;
  rows: TicketRow[];
  children?: ReactNode;
}) {
  const [open, setOpen] = useState<string | null>(null);
  const total = rows.reduce((a, r) => a + r.amount, 0);

  return (
    <section data-testid="receipt-ticket" aria-label={`Receipt: ${title}`} className="overflow-hidden rounded-3xl border border-white/5 bg-surface shadow-2xl ring-1 ring-white/10">
      <header className="space-y-4 px-6 pt-6 pb-5">
        <div className="flex items-start justify-between gap-3">
          <span className="rounded-full bg-coral/15 px-3 py-1 text-xs font-bold tracking-widest text-coral uppercase">Receipt</span>
          <span className="pt-1 font-mono text-xs tracking-wider text-muted-foreground uppercase">{shortDate(date)}</span>
        </div>
        <h2 data-testid="ticket-title" className="text-2xl leading-tight font-black tracking-tight break-words text-white">
          {title}
        </h2>
        <div className="flex items-end justify-between gap-3">
          <div>
            <p className="text-xs font-semibold text-muted-foreground">Total</p>
            <Money cents={total} currency={symbol} className="text-4xl font-black tracking-tight text-white" />
          </div>
          {rows.length > 1 && (
            <div className="space-y-1.5 text-right">
              <p className="text-xs font-semibold text-muted-foreground">Splitting with</p>
              <div className="flex justify-end">
                <AvatarStack names={rows.map((r) => r.name)} />
              </div>
            </div>
          )}
        </div>
      </header>

      {/* The tear line: a dashed rule with a notch cut out of each edge. */}
      <div aria-hidden className="relative h-6">
        <span className="absolute top-1/2 right-4 left-4 border-t-2 border-dashed border-white/10" />
        <span className="absolute top-1/2 -left-3 size-6 -translate-y-1/2 rounded-full bg-background" />
        <span className="absolute top-1/2 -right-3 size-6 -translate-y-1/2 rounded-full bg-background" />
      </div>

      <ul className="px-2 pb-2">
        {rows.map((row) => {
          const chips = parseItems(row.items);
          const expandable = chips.length > 0;
          const isOpen = open === row.name;
          const head = (
            <>
              <PersonAvatar name={row.name} />
              <div className="min-w-0 flex-1 text-left">
                <span className="block truncate text-base font-bold text-white">{row.name}</span>
                {row.note}
              </div>
              <Money cents={row.amount} currency={symbol} className="text-xl font-bold text-white" />
              {expandable && <ChevronDown aria-hidden className={cn("size-4 shrink-0 text-muted-foreground transition-transform", isOpen && "rotate-180")} />}
            </>
          );
          return (
            <li key={row.name} data-testid="ticket-row" data-person={row.name}>
              {expandable ? (
                <button
                  type="button"
                  aria-expanded={isOpen}
                  onClick={() => setOpen(isOpen ? null : row.name)}
                  className="flex min-h-16 w-full items-center gap-3 rounded-2xl px-4 py-3 transition-colors outline-none hover:bg-white/[0.03] focus-visible:ring-2 focus-visible:ring-white/30"
                >
                  {head}
                </button>
              ) : (
                <div className="flex min-h-16 items-center gap-3 px-4 py-3">{head}</div>
              )}
              <AnimatePresence initial={false}>
                {isOpen && (
                  <m.ul
                    key="items"
                    data-testid="ticket-items"
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="flex flex-wrap gap-2 overflow-hidden px-4"
                  >
                    {chips.map((c, i) => (
                      <li key={`${c.name}-${i}`} className="rounded-full bg-surface-2 px-3 py-1.5 text-sm text-zinc-200">
                        {c.name}
                        {c.share && <span className="ml-1.5 text-muted-foreground tabular-nums">{c.share}</span>}
                      </li>
                    ))}
                    <li aria-hidden className="h-2 basis-full" />
                  </m.ul>
                )}
              </AnimatePresence>
            </li>
          );
        })}
      </ul>

      {children}
    </section>
  );
}
