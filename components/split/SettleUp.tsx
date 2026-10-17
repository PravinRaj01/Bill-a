"use client";
import { useState } from "react";
import { m } from "motion/react";
import { ArrowRight, Check, Send } from "lucide-react";
import { toast } from "sonner";

import { PayerPicker } from "@/components/split/PayerPicker";
import { PaymentToggle } from "@/components/payment/PaymentToggle";
import { PersonAvatar } from "@/components/person-avatar";
import type { PaymentShare } from "@/hooks/use-payment-share";
import { Money } from "@/components/ui/money";
import { formatMoney } from "@/lib/money";
import {
  addDays,
  everyonePaidOwn,
  groupNudgeText,
  localDate,
  markPaid,
  nudgeText,
  scalePaidBy,
  settleView,
  singlePayer,
  unmarkPaid,
  withPayers,
  type Transfer,
} from "@/lib/split/settle";
import { haptic } from "@/lib/ui/haptics";
import { shareMessage, shareText } from "@/lib/ui/share";
import { cn } from "@/lib/utils";
import type { Participant, Settlement, SplitRecord } from "@/types/domain";

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
/**
 * What the card needs, so it works on the Summary (live session) and on a saved bill in History alike.
 * `change` applies an edit to the settlement and saves it; `stop` removes tracking.
 */
export interface SettleModel {
  shares: SplitRecord[];
  settlement: Settlement | null;
  symbol: string;
  billTitle: string;
  tracked: boolean;
  change: (edit: (s: Settlement) => Settlement) => void | Promise<void>;
  stop: () => void | Promise<void>;
  /** The user's payment details for messages; added only when what is owed is owed to the user. */
  payment?: PaymentShare;
  /**
   * Who was in the bill and which one is the user. When given, the card states who you are (or that the bill
   * was split for others) instead of asking; null/absent means an older bill that still needs "which one is you?".
   */
  participants?: Participant[] | null;
}

export function SettleUp({ model }: { model: SettleModel }) {
  const [custom, setCustom] = useState(false);
  // Showing the one-person chips even though several people paid ("One person paid it all").
  const [choosing, setChoosing] = useState(false);
  const { settlement, symbol, shares } = model;
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
  /**
   * One message per person with EVERYTHING they still owe, whichever of their rows was tapped. It says what
   * they owe to whom (not "your share": with several payers those differ), and describes the items only
   * when the amount is their whole share.
   */
  const nudge = (debtor: string) => {
    if (!view || !view.ok) return;
    const open = view.transfers.filter((t) => t.from === debtor && !t.paid);
    if (open.length === 0) return;
    const share = shares.find((x) => x.name === debtor);
    const wholeShare = open.length === 1 && open[0].amount === share?.amount;
    void send(
      nudgeText({
        debtor,
        title: model.billTitle,
        date: new Date(),
        currency: symbol,
        owes: open.map((t) => ({ to: payTo(t.to), amount: t.amount })),
        items: wholeShare ? share?.items : undefined,
        payment: open.every((t) => payTo(t.to) === "me") ? model.payment?.block : undefined,
      }),
    );
  };

  /** The group-chat message: one creditor ("still to pay Alex") or several ("who still owes what"). */
  const nudgeAll = () => {
    if (!view || !view.ok) return;
    const open = view.transfers.filter((t) => !t.paid);
    if (open.length === 0) return;
    const single = new Set(open.map((t) => norm(t.to))).size === 1;
    void send(
      groupNudgeText({
        title: model.billTitle,
        date: new Date(),
        currency: symbol,
        payTo: single ? payTo(open[0].to) : undefined,
        lines: open.map((t) => ({ debtor: t.from, to: t.to, amount: t.amount })),
        payment: open.every((t) => payTo(t.to) === "me") ? model.payment?.block : undefined,
      }),
    );
  };

  const toggle = (t: Transfer) => {
    if (t.paid) {
      void model.change((x) => unmarkPaid(x, t.key));
      return;
    }
    void model.change((x) => markPaid(x, t, today));
    haptic("success");
    toast(`${t.from} marked as paid`, { action: { label: "Undo", onClick: () => void model.change((x) => unmarkPaid(x, t.key)) } });
  };

  const setDue = (date: string | null) =>
    void model.change((x) => {
      const next = { ...x };
      if (date) next.dueDate = date;
      else delete next.dueDate;
      return next;
    });

  return (
    <section aria-label="Settle up" data-testid="settle-up" className="space-y-5 border-t border-white/8 p-5">
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-base font-bold text-foreground">Settle up</h3>
        {model.tracked && <span className="text-xs font-semibold text-success">Tracked in History</span>}
      </div>

      <PayerPicker
        names={names}
        paidBy={settlement.paidBy}
        total={total}
        symbol={symbol}
        onSingle={(name) => void model.change((x) => withPayers(x, singlePayer(name, total)))}
        onMulti={(paidBy) => void model.change((x) => withPayers(x, paidBy))}
        onEveryoneOwn={() => void model.change((x) => withPayers(x, everyonePaidOwn(shares)))}
        choosing={choosing}
        setChoosing={setChoosing}
      />

      {model.participants ? (
        // The bill recorded who you are, so there is nothing to ask.
        <p data-testid="me-fixed" className="flex min-h-11 items-center justify-between gap-3 px-1 text-sm">
          <span className="text-muted-foreground">{settlement.me ? "You are" : "Split for others"}</span>
          <span className="font-semibold text-foreground">{settlement.me ?? "You're not in this bill"}</span>
        </p>
      ) : (
        <label className="flex min-h-11 items-center justify-between gap-3 px-1 text-sm">
          <span className="text-muted-foreground">{settlement.me ? "You are" : "Which one is you?"}</span>
          <select
            data-testid="me-select"
            value={settlement.me ?? ""}
            onChange={(e) => void model.change((x) => ({ ...x, me: e.target.value || undefined }))}
            className="h-11 glass-inset rounded-xl px-3 text-base text-foreground"
          >
            {!settlement.me && <option value="">Choose…</option>}
            {names.map((n) => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </label>
      )}

      {view && !view.ok && (
        <div role="alert" data-testid="paid-mismatch" className="space-y-3 rounded-xl border border-warning/30 bg-warning/5 p-3 text-sm text-warning">
          <p>
            What was paid ({formatMoney(view.fronted, symbol)}) doesn&apos;t match the bill ({formatMoney(view.owed, symbol)}). That happens when the split changed after you entered who paid.
          </p>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              data-testid="scale-paid"
              onClick={() => void model.change((x) => withPayers(x, scalePaidBy(x.paidBy, view.owed)))}
              className="min-h-11 rounded-full border border-warning/40 bg-warning/10 px-4 text-sm font-semibold text-warning hover:bg-warning/20"
            >
              Scale what each paid to {formatMoney(view.owed, symbol)}
            </button>
            <button
              type="button"
              data-testid="one-paid-fix"
              onClick={() => setChoosing(true)}
              className="min-h-11 rounded-full border border-white/15 px-4 text-sm font-semibold text-foreground hover:bg-white/5"
            >
              One person paid it all
            </button>
          </div>
        </div>
      )}

      {view?.ok && (
        <>
          {view.transfers.length === 0 ? (
            <p data-testid="nothing-owed" className="rounded-xl bg-white/[0.07] p-3 text-sm text-muted-foreground">Nobody owes anybody: everyone paid their own share.</p>
          ) : (
            <>
              <div aria-live="polite" data-testid="settle-progress">
                <div className="mb-2 flex items-baseline justify-between gap-3 text-sm">
                  <span className="text-muted-foreground">
                    <Money cents={view.collected} currency={symbol} className="font-bold text-foreground" /> of{" "}
                    <Money cents={view.owed} currency={symbol} className="font-semibold text-foreground" /> collected
                  </span>
                  <span className={cn("shrink-0 text-xs font-semibold", view.status === "settled" ? "text-success" : "text-muted-foreground")}>
                    {view.status === "settled" ? "✓ All settled" : `${view.paidCount} of ${view.transfers.length} paid`}
                  </span>
                </div>
                <div role="progressbar" aria-label="Collected" aria-valuemin={0} aria-valuemax={view.owed} aria-valuenow={view.collected} className="h-2.5 overflow-hidden rounded-full bg-white/10">
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
                    className={cn("flex items-center gap-2 rounded-2xl bg-white/[0.07] p-2 pl-3", t.paid && "opacity-60")}
                  >
                    {/* One avatar (who owes); the sentence names both people and may wrap, so it is never cut off. */}
                    <div className="flex min-w-0 flex-1 items-center gap-3">
                      <PersonAvatar name={t.from} compact className="size-9 shrink-0" />
                      <div className="min-w-0">
                        <p className={cn("text-sm leading-snug font-semibold text-foreground", t.paid && "line-through")}>
                          {t.from} <ArrowRight className="inline size-3.5 text-muted-foreground" aria-hidden /> {t.to}
                          <span className="sr-only"> (pays)</span>
                        </p>
                        <Money cents={t.amount} currency={symbol} className="text-lg font-bold text-foreground" />
                      </div>
                    </div>
                    {!t.paid && (
                      <button
                        type="button"
                        aria-label={`Nudge ${t.from}`}
                        onClick={() => nudge(t.from)}
                        className="flex size-11 shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-white/5 hover:text-foreground"
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
                        t.paid ? "border-success bg-success text-background" : "border-white/15 text-muted-foreground hover:border-white/40 hover:text-foreground",
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
                  onClick={nudgeAll}
                  className="flex min-h-11 w-full items-center justify-center gap-2 rounded-xl border border-white/10 text-sm font-semibold text-foreground transition-colors hover:bg-white/5"
                >
                  <Send className="size-4" /> Nudge everyone who owes
                </button>
              )}
              {model.payment && view.status !== "settled" && <PaymentToggle payment={model.payment} />}
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
                  <button key={label} type="button" aria-pressed={on} onClick={() => setDue(on ? null : date)} className={cn(chip, on ? "border-coral bg-coral/15 text-coral" : "border-white/10 bg-white/5 text-foreground/80 hover:bg-white/10")}>
                    {label}
                  </button>
                );
              })}
              <span className={cn(chip, "relative cursor-pointer", custom ? "border-coral bg-coral/15 text-coral" : "border-white/10 bg-white/5 text-foreground/80 hover:bg-white/10", "flex items-center")}>
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
                <button type="button" onClick={() => { setCustom(false); setDue(null); }} className={cn(chip, "border-transparent text-muted-foreground hover:text-foreground")}>
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

      {model.tracked && (
        <button type="button" onClick={() => void model.stop()} className="min-h-11 px-1 text-sm text-muted-foreground underline-offset-4 hover:text-foreground hover:underline">
          Stop tracking this bill
        </button>
      )}
    </section>
  );
}
