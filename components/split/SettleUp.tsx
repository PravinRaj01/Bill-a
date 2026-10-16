"use client";
import { useState } from "react";
import { m } from "motion/react";
import { ArrowRight, Check, Send } from "lucide-react";
import { toast } from "sonner";

import { PayerPicker } from "@/components/split/PayerPicker";
import { PersonAvatar } from "@/components/person-avatar";
import { Money } from "@/components/ui/money";
import type { SplitSession } from "@/hooks/use-split-session";
import { formatMoney } from "@/lib/money";
import {
  addDays,
  groupNudgeText,
  localDate,
  markPaid,
  nudgeText,
  settleView,
  singlePayer,
  unmarkPaid,
  withPayers,
  type Transfer,
} from "@/lib/split/settle";
import { haptic } from "@/lib/ui/haptics";
import { shareMessage, shareText } from "@/lib/ui/share";
import { cn } from "@/lib/utils";

const norm = (s: string) => s.trim().toLowerCase();
const chip = "min-h-11 rounded-full border px-4 text-sm font-semibold transition-colors outline-none focus-visible:ring-2 focus-visible:ring-white/30";

function dueLabel(due: string, dueIn: number): { text: string; tone: string } {
  const [y, mo, d] = due.split("-").map(Number);
  const day = new Date(y, mo - 1, d).toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short" });
  if (dueIn < 0) return { text: `Due ${day} · overdue by ${-dueIn} ${dueIn === -1 ? "day" : "days"}`, tone: "text-coral" };
  if (dueIn === 0) return { text: `Due today · ${day}`, tone: "text-warning" };
  return { text: `Due ${day} · in ${dueIn} ${dueIn === 1 ? "day" : "days"}`, tone: "text-muted-foreground" };
}

// Settle-up on the Summary: who fronted the bill, who owes whom, who has paid back. The transfers are
// worked out from the shares (lib/split/settle.ts); nothing is SAVED with the bill until the first
// real interaction (Mark paid, a different payer, a due date), so just looking never turns a dinner
// everyone paid for in cash into an open debt in History.
export function SettleUp({ s }: { s: SplitSession }) {
  const [custom, setCustom] = useState(false);
  const { settlement, symbol, finalSplit: shares } = s;
  if (!settlement || shares.length < 2) return null;

  const today = localDate();
  const view = settleView(shares, settlement, today);
  const names = shares.map((x) => x.name);
  const total = shares.reduce((a, b) => a + b.amount, 0);
  const meKey = settlement.me ? norm(settlement.me) : null;
  const payTo = (name: string) => (meKey && norm(name) === meKey ? "me" : name);

  const send = async (text: string) => {
    const msg = shareMessage(await shareText(text));
    if (msg) toast(msg);
  };
  const nudge = (t: Transfer) =>
    send(nudgeText({ debtor: t.from, payTo: payTo(t.to), title: s.billTitle, date: new Date(), amount: t.amount, currency: symbol, items: shares.find((x) => x.name === t.from)?.items }));

  const toggle = (t: Transfer) => {
    if (t.paid) {
      void s.changeSettlement((x) => unmarkPaid(x, t.key));
      return;
    }
    void s.changeSettlement((x) => markPaid(x, t, today));
    haptic("success");
    toast(`${t.from} marked as paid`, { action: { label: "Undo", onClick: () => void s.changeSettlement((x) => unmarkPaid(x, t.key)) } });
  };

  const setDue = (date: string | null) =>
    void s.changeSettlement((x) => {
      const next = { ...x };
      if (date) next.dueDate = date;
      else delete next.dueDate;
      return next;
    });

  return (
    <section aria-label="Settle up" data-testid="settle-up" className="space-y-5 border-t border-white/5 p-5">
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-base font-bold text-white">Settle up</h3>
        {s.settleTracked && <span className="text-xs font-semibold text-success">Tracked in History</span>}
      </div>

      <PayerPicker
        names={names}
        paidBy={settlement.paidBy}
        total={total}
        symbol={symbol}
        onSingle={(name) => void s.changeSettlement((x) => withPayers(x, singlePayer(name, total)))}
        onMulti={(paidBy) => void s.changeSettlement((x) => withPayers(x, paidBy))}
      />

      <label className="flex min-h-11 items-center justify-between gap-3 px-1 text-sm">
        <span className="text-muted-foreground">{settlement.me ? "You are" : "Which one is you?"}</span>
        <select
          data-testid="me-select"
          value={settlement.me ?? ""}
          onChange={(e) => void s.changeSettlement((x) => ({ ...x, me: e.target.value || undefined }))}
          className="h-11 rounded-xl border border-white/10 bg-surface-2 px-3 text-base text-white"
        >
          {!settlement.me && <option value="">Choose…</option>}
          {names.map((n) => (
            <option key={n} value={n}>{n}</option>
          ))}
        </select>
      </label>

      {view && !view.ok && (
        <p role="alert" className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-3 text-sm text-amber-200">
          What was paid ({formatMoney(view.fronted, symbol)}) doesn&apos;t match the bill ({formatMoney(view.owed, symbol)}). Change who paid to fix it.
        </p>
      )}

      {view?.ok && (
        <>
          {view.transfers.length === 0 ? (
            <p data-testid="nothing-owed" className="rounded-xl bg-surface-2/60 p-3 text-sm text-muted-foreground">Nobody owes anybody: everyone paid their own share.</p>
          ) : (
            <>
              <div aria-live="polite" data-testid="settle-progress">
                <div className="mb-2 flex items-baseline justify-between gap-3 text-sm">
                  <span className="text-muted-foreground">
                    <Money cents={view.collected} currency={symbol} className="font-bold text-white" /> of{" "}
                    <Money cents={view.owed} currency={symbol} className="font-semibold text-white" /> collected
                  </span>
                  <span className={cn("shrink-0 text-xs font-semibold", view.status === "settled" ? "text-success" : "text-muted-foreground")}>
                    {view.status === "settled" ? "✓ All settled" : `${view.paidCount} of ${view.transfers.length} paid`}
                  </span>
                </div>
                <div role="progressbar" aria-label="Collected" aria-valuemin={0} aria-valuemax={view.owed} aria-valuenow={view.collected} className="h-2.5 overflow-hidden rounded-full bg-surface-2">
                  <m.div className="h-full rounded-full bg-success" initial={false} animate={{ width: `${(view.collected / view.owed) * 100}%` }} transition={{ type: "spring", stiffness: 300, damping: 36 }} />
                </div>
              </div>

              <ul className="space-y-2">
                {[...view.transfers].sort((a, b) => Number(a.paid) - Number(b.paid)).map((t) => (
                  <m.li
                    key={t.key}
                    layout
                    data-testid="transfer"
                    data-paid={t.paid ? "true" : undefined}
                    transition={{ type: "spring", stiffness: 500, damping: 40 }}
                    className={cn("flex items-center gap-2 rounded-2xl bg-surface-2/60 p-2 pl-3", t.paid && "opacity-60")}
                  >
                    {/* One avatar (who owes); the sentence names both people and may wrap, so it is never cut off. */}
                    <div className="flex min-w-0 flex-1 items-center gap-3">
                      <PersonAvatar name={t.from} compact className="size-9 shrink-0" />
                      <div className="min-w-0">
                        <p className={cn("text-sm leading-snug font-semibold text-white", t.paid && "line-through")}>
                          {t.from} <ArrowRight className="inline size-3.5 text-muted-foreground" aria-hidden /> {t.to}
                          <span className="sr-only"> (pays)</span>
                        </p>
                        <Money cents={t.amount} currency={symbol} className="text-lg font-bold text-white" />
                      </div>
                    </div>
                    {!t.paid && (
                      <button
                        type="button"
                        aria-label={`Nudge ${t.from}`}
                        onClick={() => void nudge(t)}
                        className="flex size-11 shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-white/5 hover:text-white"
                      >
                        <Send className="size-4" />
                      </button>
                    )}
                    <button
                      type="button"
                      aria-pressed={t.paid}
                      aria-label={t.paid ? `Mark ${t.from} as not paid` : `Mark ${t.from} as paid`}
                      onClick={() => toggle(t)}
                      className={cn(
                        "flex size-11 shrink-0 items-center justify-center rounded-full border transition-colors",
                        t.paid ? "border-success bg-success text-black" : "border-white/15 text-zinc-400 hover:border-white/40 hover:text-white",
                      )}
                    >
                      <Check className="size-5" strokeWidth={3} />
                    </button>
                  </m.li>
                ))}
              </ul>

              {view.status !== "settled" && (
                <button
                  type="button"
                  data-testid="nudge-all"
                  onClick={() => {
                    const open = view.transfers.filter((t) => !t.paid);
                    const creditors = new Set(open.map((t) => norm(t.to)));
                    const single = creditors.size === 1;
                    void send(
                      groupNudgeText({
                        title: s.billTitle,
                        date: new Date(),
                        payTo: single ? payTo(open[0].to) : "whoever fronted it",
                        currency: symbol,
                        lines: open.map((t) => ({ debtor: single ? t.from : `${t.from} → ${t.to}`, amount: t.amount })),
                      }),
                    );
                  }}
                  className="flex min-h-11 w-full items-center justify-center gap-2 rounded-xl border border-white/10 text-sm font-semibold text-zinc-200 transition-colors hover:bg-white/5"
                >
                  <Send className="size-4" /> Nudge everyone who owes
                </button>
              )}
            </>
          )}

          <div className="space-y-2">
            <p className="px-1 text-xs font-semibold text-muted-foreground">Remind them by</p>
            <div className="flex flex-wrap gap-2">
              {[
                ["Tomorrow", 1],
                ["In 3 days", 3],
                ["In a week", 7],
              ].map(([label, n]) => {
                const date = addDays(today, n as number);
                const on = settlement.dueDate === date;
                return (
                  <button key={label} type="button" aria-pressed={on} onClick={() => setDue(on ? null : date)} className={cn(chip, on ? "border-coral bg-coral/15 text-coral" : "border-white/10 bg-white/5 text-zinc-300 hover:bg-white/10")}>
                    {label}
                  </button>
                );
              })}
              <span className={cn(chip, "relative cursor-pointer", custom ? "border-coral bg-coral/15 text-coral" : "border-white/10 bg-white/5 text-zinc-300 hover:bg-white/10", "flex items-center")}>
                Pick a date
                <input
                  type="date"
                  aria-label="Pick a due date"
                  data-testid="due-date"
                  min={today}
                  value={settlement.dueDate ?? ""}
                  onClick={(e) => e.currentTarget.showPicker?.()}
                  onChange={(e) => {
                    setCustom(!!e.target.value);
                    setDue(e.target.value || null);
                  }}
                  className="absolute inset-0 size-full cursor-pointer opacity-0"
                />
              </span>
              {settlement.dueDate && (
                <button type="button" onClick={() => { setCustom(false); setDue(null); }} className={cn(chip, "border-transparent text-muted-foreground hover:text-white")}>
                  Clear
                </button>
              )}
            </div>
            {settlement.dueDate && view.dueIn !== null && view.status !== "settled" && (
              <p data-testid="due-label" className={cn("px-1 text-sm font-semibold", dueLabel(settlement.dueDate, view.dueIn).tone)}>
                {dueLabel(settlement.dueDate, view.dueIn).text}
              </p>
            )}
          </div>
        </>
      )}

      {s.settleTracked && (
        <button type="button" onClick={() => void s.stopTracking()} className="min-h-11 px-1 text-sm text-muted-foreground underline-offset-4 hover:text-white hover:underline">
          Stop tracking this bill
        </button>
      )}
    </section>
  );
}
