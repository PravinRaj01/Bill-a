"use client";
import { useState } from "react";
import { NumberFlowGroup } from "@number-flow/react";
import { Check, Lock, LockOpen, RotateCcw } from "lucide-react";
import { toast } from "sonner";

import { PersonAvatar } from "@/components/person-avatar";
import { Money } from "@/components/ui/money";
import { Slider } from "@/components/ui/slider";
import type { ShareEditor } from "@/hooks/use-share-editor";
import { centsToInput, formatMoney, parseMoneyInput } from "@/lib/money";
import { haptic } from "@/lib/ui/haptics";
import { personColor } from "@/lib/ui/person-color";
import { centsForPercent, percentOf } from "@/lib/split/sliders";
import { cn } from "@/lib/utils";

// One row per person: who, their share as a percent and as money (tap either to type an exact
// number, which locks that person), a lock, and a slider in the person's colour. Moving a slider
// pushes the unlocked others; the footer always reads 100% because the shares are cents that add
// up to the bill by construction (lib/split/sliders.ts).

/** A value that shows as a button and turns into a number field when tapped. Enter/blur saves, Escape cancels. */
function InlineValue({
  label,
  display,
  initial,
  onCommit,
  disabled,
  className,
  width = "w-20",
}: {
  label: string;
  display: React.ReactNode;
  initial: string;
  onCommit: (raw: string) => void;
  disabled?: boolean;
  className?: string;
  width?: string;
}) {
  const [editing, setEditing] = useState(false);
  if (editing) {
    return (
      <input
        autoFocus
        aria-label={label}
        inputMode="decimal"
        defaultValue={initial}
        onFocus={(e) => e.currentTarget.select()}
        onBlur={(e) => {
          setEditing(false);
          onCommit(e.currentTarget.value);
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter") e.currentTarget.blur();
          if (e.key === "Escape") {
            e.currentTarget.value = initial;
            e.currentTarget.blur();
          }
        }}
        className={cn("h-10 rounded-lg border border-white/15 bg-black/40 px-2 text-right text-base tabular-nums text-white outline-none focus:border-white/40", width)}
      />
    );
  }
  return (
    <button
      type="button"
      aria-label={`${label}, tap to type`}
      disabled={disabled}
      onClick={() => setEditing(true)}
      className={cn("min-h-10 rounded-lg px-2 text-right tabular-nums transition-colors enabled:hover:bg-white/5 disabled:cursor-default", className)}
    >
      {display}
    </button>
  );
}

export function SharesEditor({
  people,
  symbol,
  editor,
  readOnly = false,
}: {
  people: string[];
  symbol: string;
  editor: ShareEditor;
  /** Just show everyone's share (the Equal preview): no sliders, locks or typing. */
  readOnly?: boolean;
}) {
  const { total, amounts, locked } = editor;
  // Half-percent steps: an arrow key moves 0.5%, Shift/PageUp 5%.
  const step = Math.max(1, Math.round(total / 200));

  const onDrag = (i: number, cents: number) => {
    const before = percentOf(amounts[i], total);
    editor.drag(i, cents);
    const after = percentOf(Math.min(cents, editor.maxFor(i)), total);
    if (Math.floor(after / 5) !== Math.floor(before / 5)) haptic("selection");
  };

  const onType = (i: number, cents: number) => {
    if (!editor.canMove(i)) {
      toast("Unlock someone else first — their share has to absorb the change.");
      return;
    }
    editor.type(i, cents);
  };

  const reset = () => {
    const undo = editor.reset();
    toast("Shares reset to equal", { action: { label: "Undo", onClick: undo } });
  };

  return (
    <NumberFlowGroup>
      <ul data-testid="shares-editor" className="space-y-2">
        {people.map((name, i) => {
          const cents = amounts[i];
          const pct = percentOf(cents, total);
          const frozen = readOnly || locked[i] || !editor.canMove(i);
          return (
            <li key={name} data-testid="share-row" data-locked={locked[i] ? "true" : undefined} className="rounded-2xl bg-surface-2/60 px-3 pt-2 pb-0.5">
              {/* Line 1: who, their amount, the lock. Line 2: the slider with its percent beside it. */}
              <div className="flex items-center gap-2">
                <PersonAvatar name={name} />
                <span className="min-w-0 flex-1 truncate text-base font-semibold text-white">{name}</span>
                {readOnly ? (
                  <Money cents={cents} currency={symbol} className="text-right text-base font-bold text-white" />
                ) : (
                  <InlineValue
                    label={`${name}'s amount`}
                    display={<Money cents={cents} currency={symbol} className="text-base font-bold text-white" />}
                    initial={centsToInput(cents)}
                    disabled={locked[i]}
                    width="w-24"
                    onCommit={(raw) => onType(i, parseMoneyInput(raw))}
                  />
                )}
                {!readOnly && (
                  <button
                    type="button"
                    aria-label={`${locked[i] ? "Unlock" : "Lock"} ${name}'s share`}
                    aria-pressed={locked[i]}
                    onClick={() => editor.toggleLock(i)}
                    className={cn(
                      "flex size-11 shrink-0 items-center justify-center rounded-full transition-colors",
                      locked[i] ? "bg-coral/15 text-coral" : "text-zinc-500 hover:bg-white/5 hover:text-white",
                    )}
                  >
                    {locked[i] ? <Lock className="size-4" /> : <LockOpen className="size-4" />}
                  </button>
                )}
              </div>
              {readOnly ? (
                <p className="pb-2 pl-11 text-sm tabular-nums text-muted-foreground">{pct}% of the bill</p>
              ) : (
                <div className="flex items-center gap-1">
                  <Slider
                    aria-label={`${name}'s share`}
                    valueText={`${name}, ${pct} percent, ${formatMoney(cents, symbol)}`}
                    value={[cents]}
                    min={0}
                    max={Math.max(total, 1)}
                    step={step}
                    disabled={frozen}
                    onValueChange={([v]) => onDrag(i, v)}
                    style={{ ["--slider-color" as string]: personColor(name) }}
                  />
                  <InlineValue
                    label={`${name}'s percent`}
                    display={<span className="text-sm text-muted-foreground">{pct}%</span>}
                    initial={String(pct)}
                    disabled={locked[i]}
                    width="w-16"
                    className="w-14 shrink-0"
                    onCommit={(raw) => {
                      const n = parseFloat(raw.replace(",", "."));
                      if (Number.isFinite(n)) onType(i, centsForPercent(n, total));
                    }}
                  />
                </div>
              )}
            </li>
          );
        })}
      </ul>

      <div className="mt-3 flex items-center justify-between gap-3 px-1">
        <div className="flex items-center gap-2 text-sm" aria-live="polite" data-testid="shares-total">
          <Money cents={amounts.reduce((a, b) => a + b, 0)} currency={symbol} className="font-bold text-white" />
          <span className="flex items-center gap-1 font-semibold text-success">
            <Check className="size-3.5" /> 100%
          </span>
        </div>
        {!readOnly && people.length > 1 && (
          <button
            type="button"
            onClick={reset}
            className="flex min-h-11 items-center gap-1.5 rounded-full px-3 text-sm font-semibold text-muted-foreground transition-colors hover:bg-white/5 hover:text-white"
          >
            <RotateCcw className="size-3.5" /> Reset to equal
          </button>
        )}
      </div>
    </NumberFlowGroup>
  );
}
