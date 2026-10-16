"use client";
import { useState } from "react";

import { PersonAvatar } from "@/components/person-avatar";
import { Button } from "@/components/ui/button";
import { centsToInput, formatMoney, parseMoneyInput } from "@/lib/money";
import { haptic } from "@/lib/ui/haptics";
import { cn } from "@/lib/utils";
import type { Cents, Settlement } from "@/types/domain";

// "Paid by": one person fronted the whole bill (the usual case: a row of avatar chips, one tap), or
// several people each paid part of it ("Several people paid": amounts that must add up to the bill).

function Chips({ names, selected, onPick }: { names: string[]; selected: string; onPick: (name: string) => void }) {
  // A radio group: one tab stop, arrow keys move the selection.
  const onKeyDown = (e: React.KeyboardEvent) => {
    const i = names.indexOf(selected);
    const step = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 0;
    if (!step) return;
    e.preventDefault();
    onPick(names[(i + step + names.length) % names.length]);
  };
  return (
    <div role="radiogroup" aria-label="Paid by" onKeyDown={onKeyDown} className="-mx-1 flex gap-2 overflow-x-auto px-1 pb-1">
      {names.map((name) => {
        const on = name === selected;
        return (
          <button
            key={name}
            type="button"
            role="radio"
            aria-checked={on}
            tabIndex={on ? 0 : -1}
            onClick={() => {
              if (!on) haptic("selection");
              onPick(name);
            }}
            className={cn(
              "flex min-h-11 shrink-0 items-center gap-2 rounded-full border py-1 pr-4 pl-1 text-sm font-semibold transition-colors outline-none focus-visible:ring-2 focus-visible:ring-white/30",
              on ? "border-coral bg-coral/15 text-white" : "border-white/10 bg-white/5 text-zinc-300 hover:bg-white/10",
            )}
          >
            <PersonAvatar name={name} compact className="size-8" />
            {name}
          </button>
        );
      })}
    </div>
  );
}

function MultiEditor({
  names,
  total,
  symbol,
  initial,
  onApply,
  onCancel,
}: {
  names: string[];
  total: Cents;
  symbol: string;
  initial: Settlement["paidBy"];
  onApply: (paidBy: Settlement["paidBy"]) => void;
  onCancel: () => void;
}) {
  const [draft, setDraft] = useState<Record<string, string>>(() =>
    Object.fromEntries(names.map((n) => [n, initial.some((p) => p.name === n && p.amount > 0) ? centsToInput(initial.find((p) => p.name === n)!.amount) : ""])),
  );
  const cents = (n: string) => parseMoneyInput(draft[n] ?? "");
  const entered = names.reduce((a, n) => a + cents(n), 0);
  const left = total - entered;
  const ok = left === 0 && entered > 0;

  return (
    <div data-testid="multi-payer" className="space-y-3 rounded-2xl bg-surface-2/60 p-3">
      <p className="px-1 text-sm text-muted-foreground">How much did each person pay?</p>
      <ul className="space-y-2">
        {names.map((n) => (
          <li key={n} className="flex items-center gap-2">
            <PersonAvatar name={n} compact className="size-8" />
            <span className="min-w-0 flex-1 truncate text-sm font-semibold text-white">{n}</span>
            {left > 0 && cents(n) === 0 && (
              <button type="button" onClick={() => setDraft({ ...draft, [n]: centsToInput(left) })} className="min-h-9 rounded-full px-3 text-xs font-semibold text-coral hover:bg-coral/10">
                Rest
              </button>
            )}
            <input
              aria-label={`${n} paid`}
              inputMode="decimal"
              placeholder="0.00"
              value={draft[n]}
              onChange={(e) => setDraft({ ...draft, [n]: e.target.value.replace(/[^0-9.,]/g, "").slice(0, 10) })}
              className="h-11 w-24 rounded-lg border border-white/10 bg-black/30 px-2 text-right text-base tabular-nums text-white outline-none focus:border-white/40"
            />
          </li>
        ))}
      </ul>
      <p
        role="status"
        data-testid="multi-left"
        className={cn("px-1 text-sm font-semibold tabular-nums", left === 0 ? "text-success" : left > 0 ? "text-warning" : "text-red-400")}
      >
        {left === 0 ? `✓ Adds up to ${formatMoney(total, symbol)}` : left > 0 ? `${formatMoney(left, symbol)} left to assign` : `${formatMoney(-left, symbol)} too much`}
      </p>
      <div className="flex gap-2">
        <Button type="button" variant="outline" className="h-11 flex-1 rounded-xl border-white/10 text-sm text-zinc-300" onClick={onCancel}>
          Cancel
        </Button>
        <Button
          type="button"
          data-testid="multi-apply"
          disabled={!ok}
          className="h-11 flex-1 rounded-xl bg-coral text-sm font-bold text-coral-foreground hover:bg-coral/90"
          onClick={() => onApply(names.filter((n) => cents(n) > 0).map((n) => ({ name: n, amount: cents(n) })))}
        >
          Apply
        </Button>
      </div>
    </div>
  );
}

export function PayerPicker({
  names,
  paidBy,
  total,
  symbol,
  onSingle,
  onMulti,
}: {
  names: string[];
  paidBy: Settlement["paidBy"];
  total: Cents;
  symbol: string;
  onSingle: (name: string) => void;
  onMulti: (paidBy: Settlement["paidBy"]) => void;
}) {
  const [editing, setEditing] = useState(false);
  const multi = paidBy.length > 1;

  if (editing) {
    return (
      <MultiEditor
        names={names}
        total={total}
        symbol={symbol}
        initial={paidBy}
        onCancel={() => setEditing(false)}
        onApply={(p) => {
          setEditing(false);
          if (p.length === 1) onSingle(p[0].name);
          else onMulti(p);
        }}
      />
    );
  }

  return (
    <div className="space-y-2">
      <p className="px-1 text-xs font-semibold text-muted-foreground">Paid by</p>
      {multi ? (
        <div data-testid="multi-summary" className="flex flex-wrap items-center gap-2 rounded-2xl bg-surface-2/60 p-3 text-sm">
          {paidBy.map((p) => (
            <span key={p.name} className="flex items-center gap-1.5 rounded-full bg-white/5 py-1 pr-3 pl-1 font-semibold text-white">
              <PersonAvatar name={p.name} compact className="size-7" />
              {p.name} <span className="tabular-nums text-muted-foreground">{formatMoney(p.amount, symbol)}</span>
            </span>
          ))}
        </div>
      ) : (
        <Chips names={names} selected={paidBy[0]?.name ?? ""} onPick={onSingle} />
      )}
      <button type="button" onClick={() => setEditing(true)} className="min-h-11 px-1 text-sm font-semibold text-muted-foreground underline-offset-4 hover:text-white hover:underline">
        {multi ? "Change who paid" : "Several people paid"}
      </button>
    </div>
  );
}
