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
    <m.div
      data-testid="receipt-ticket"
      role="region"
      aria-label={`Receipt: ${title}`}
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
      className="drop-shadow-[0_18px_28px_rgb(0_0_0/0.45)]"
    >
      {/* Top of the ticket: what it is, when, how much, who. */}
      <header className="ticket-top space-y-4 rounded-t-3xl bg-surface bg-[linear-gradient(180deg,rgb(255_255_255/0.06),transparent_70%)] px-6 pt-6 pb-8">
        <div className="flex items-center justify-between gap-3">
          <span className="rounded-full bg-coral/15 px-3 py-1 text-xs font-bold tracking-widest text-coral uppercase">Receipt</span>
          <span className="type-eyebrow">{shortDate(date)}</span>
        </div>
        <h2 data-testid="ticket-title" className="text-2xl leading-tight font-black tracking-tight break-words text-foreground">
          {title}
        </h2>
        <div className="flex items-end justify-between gap-3">
          <div>
            <p className="type-eyebrow">Total</p>
            <Money cents={total} currency={symbol} className="text-4xl font-black tracking-tight text-foreground" />
          </div>
          {rows.length > 1 && (
            <div className="space-y-1.5 text-right">
              <p className="type-eyebrow">Splitting with</p>
              <div className="flex justify-end">
                <AvatarStack names={rows.map((r) => r.name)} />
              </div>
            </div>
          )}
        </div>
      </header>

      {/* The body is clipped twice: notches at the top (the tear line), a zigzag torn edge at the bottom. */}
      <div className="ticket-torn">
        <div className="ticket-notched relative bg-surface pt-5 pb-3">
          <div aria-hidden className="absolute top-0 right-5 left-5 border-t-2 border-dashed border-white/15" />
          <ul className="px-2 pb-2">
        {rows.map((row) => {
          const chips = parseItems(row.items);
          const expandable = chips.length > 0;
          const isOpen = open === row.name;
          const head = (
            <>
              <PersonAvatar name={row.name} />
              <div className="min-w-0 flex-1 text-left sm:flex-none">
                <span className="block truncate text-base font-bold text-foreground">{row.name}</span>
                {row.note}
              </div>
              {/* The dotted leader from the name to the amount, as on a till receipt. */}
              <span aria-hidden className="mb-2 hidden min-w-3 flex-1 self-end border-b-2 border-dotted border-white/15 sm:block" />
              <Money cents={row.amount} currency={symbol} className="text-xl font-bold text-foreground" />
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
                      <li key={`${c.name}-${i}`} className="rounded-full bg-surface-2 px-3 py-1.5 text-sm text-foreground">
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

          <div aria-hidden className="mx-6 mt-2 mb-4 space-y-1.5 text-center">
            <div className="ticket-barcode" />
            <p className="font-mono text-[10px] tracking-[0.3em] text-muted-foreground uppercase">Split with Bill.a</p>
          </div>
        </div>
      </div>
    </m.div>
  );
}
