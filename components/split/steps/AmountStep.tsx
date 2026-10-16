"use client";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { formatMoney, parseMoneyInput } from "@/lib/money";
import { MAX_QUICK_CENTS, quickTax } from "@/lib/split/quick";
import type { SplitSession } from "@/hooks/use-split-session";
import { cn } from "@/lib/utils";

const chip = "min-h-11 rounded-full border px-4 text-sm font-semibold transition-colors";

// Quick split, step 1: how much? A big calculator-style amount, an optional name for the bill, and
// optional Service 10% / SST 6% chips for a restaurant total typed in without tax. No receipt, no scan.
export function AmountStep({ s }: { s: SplitSession }) {
  const [text, setText] = useState("");
  const [title, setTitle] = useState("");
  const [service, setService] = useState(false);
  const [sst, setSst] = useState(false);

  const cents = parseMoneyInput(text);
  const tooBig = cents > MAX_QUICK_CENTS;
  const q = quickTax(Math.min(cents, MAX_QUICK_CENTS), { service, sst });
  const ok = cents > 0 && !tooBig;

  const go = () => {
    if (ok) s.startQuick({ cents, title, taxCents: q.tax });
  };

  return (
    <div className="space-y-6">
      <div className="space-y-1">
        <h2 className="text-2xl font-bold tracking-tight text-white">How much?</h2>
        <p className="text-sm text-muted-foreground">Step 1 of 3 · no receipt needed</p>
      </div>

      <div className="rounded-3xl border border-white/5 bg-surface p-5 shadow-2xl">
        <label htmlFor="quick-amount" className="sr-only">Amount to split</label>
        <div className="flex items-baseline justify-center gap-2 py-4">
          <span className="text-2xl font-semibold text-muted-foreground">RM</span>
          <input
            id="quick-amount"
            data-testid="quick-amount"
            autoFocus
            inputMode="decimal"
            enterKeyHint="next"
            placeholder="0.00"
            value={text}
            // digits and one decimal separator only: nothing else can reach the total
            onChange={(e) => setText(e.target.value.replace(/[^0-9.,]/g, "").slice(0, 10))}
            onKeyDown={(e) => e.key === "Enter" && go()}
            // sized to what is typed so "RM" and the number stay together in the middle
            style={{ width: `${Math.max(text.length, 4)}ch` }}
            className="min-w-0 max-w-[70%] bg-transparent text-center text-5xl font-bold tabular-nums tracking-tight text-white outline-none placeholder:text-zinc-700"
          />
        </div>
        {tooBig && <p role="alert" className="text-center text-sm text-warning">That&apos;s more than RM99,999.99. Try a smaller amount.</p>}

        <div className="mt-4 space-y-4">
          <Input
            aria-label="What is it for? (optional)"
            placeholder="What's it for? (optional) e.g. Taxi, Dinner"
            value={title}
            maxLength={60}
            onChange={(e) => setTitle(e.target.value)}
            className="h-12 border-white/5 bg-surface-2 text-base text-white"
          />

          <div>
            <p className="mb-2 px-1 text-xs font-semibold text-muted-foreground">Add to the amount (optional)</p>
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                aria-pressed={service}
                onClick={() => setService(!service)}
                className={cn(chip, service ? "border-coral bg-coral/15 text-coral" : "border-white/10 bg-white/5 text-zinc-300 hover:bg-white/10")}
              >
                + Service charge 10%
              </button>
              <button
                type="button"
                aria-pressed={sst}
                onClick={() => setSst(!sst)}
                className={cn(chip, sst ? "border-coral bg-coral/15 text-coral" : "border-white/10 bg-white/5 text-zinc-300 hover:bg-white/10")}
              >
                + SST 6%
              </button>
            </div>
          </div>

          {q.tax > 0 && ok && (
            <div data-testid="quick-breakdown" className="space-y-1 rounded-xl bg-surface-2 p-3 text-sm text-muted-foreground">
              {q.service > 0 && <p className="flex justify-between"><span>Service charge 10%</span><span className="tabular-nums">{formatMoney(q.service)}</span></p>}
              {q.sst > 0 && <p className="flex justify-between"><span>SST 6%{q.service > 0 ? " (on amount + service)" : ""}</span><span className="tabular-nums">{formatMoney(q.sst)}</span></p>}
              <p className="flex justify-between border-t border-white/5 pt-1 font-semibold text-white"><span>Total to split</span><span data-testid="quick-total" className="tabular-nums">{formatMoney(q.total)}</span></p>
            </div>
          )}
        </div>

        <Button
          type="button"
          data-testid="quick-continue"
          className="mt-5 h-12 w-full rounded-xl bg-coral text-base font-bold text-coral-foreground shadow-[var(--coral-glow)] hover:bg-coral/90"
          disabled={!ok}
          onClick={go}
        >
          Continue
        </Button>
      </div>
    </div>
  );
}
