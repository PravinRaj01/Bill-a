"use client";
import { useState } from "react";
import { ChevronDown, ChevronUp, Loader2, ScrollText, Share2, Sparkles } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ClarifyChips } from "@/components/ai/ClarifyChips";
import { CrossCheck } from "@/components/ai/CrossCheck";
import { ReceiptTicket, type TicketRow } from "@/components/receipt/ReceiptTicket";
import { AdjustShares } from "@/components/split/AdjustShares";
import { SettleUp } from "@/components/split/SettleUp";
import { usePaymentShare } from "@/hooks/use-payment-share";
import { allOwedToMe, localDate, positionNote, positions, settleView, summaryText } from "@/lib/split/settle";
import { shareMessage, shareText } from "@/lib/ui/share";
import { toast } from "sonner";
import { Segmented } from "@/components/ui/segmented";
import { CASH_UNITS } from "@/lib/split/cashRound";
import { formatMoney } from "@/lib/money";
import type { SplitSession } from "@/hooks/use-split-session";

export function SummaryStep({ s }: { s: SplitSession }) {
  const [showReasoning, setShowReasoning] = useState(false);
  const [modifyText, setModifyText] = useState("");
  const { symbol } = s;
  const payment = usePaymentShare();

  const apply = () => {
    const text = modifyText.trim();
    if (!text) return;
    setModifyText("");
    s.modify(text);
  };

  // Settle-up state of this bill (null until there are two people to settle between).
  const settleNow = s.settlement && s.finalSplit.length > 1 ? settleView(s.finalSplit, s.settlement, localDate()) : null;
  const positionOf = new Map(
    settleNow?.ok && s.settlement ? positions(s.finalSplit, s.settlement.paidBy).map((p) => [p.name, p] as const) : [],
  );

  const share = () => {
    // "Pay me" only goes out when what is still owed is owed to the user.
    const text = summaryText({
      currency: symbol,
      shares: s.finalSplit,
      settlement: s.settlement,
      view: settleNow,
      payment: allOwedToMe(settleNow, s.settlement?.me) ? payment.block : undefined,
    });
    void shareText(text).then((o) => {
      const msg = shareMessage(o);
      if (msg) toast(msg);
    });
  };

  const notes = (
    <>
      {s.clarify && <ClarifyChips chips={s.clarify.chips} hasPreview={!!s.clarify.preview} onAction={s.handleClarify} />}
      {s.disagree && <CrossCheck aiName={s.disagree.tier === "groq" ? "Groq" : "Gemini"} ai={s.disagree.ai.result} rules={s.disagree.rules.result} symbol={symbol} onPick={s.pickReading} />}
      {s.splitNote && !s.clarify && <p data-testid="split-note" className="text-xs text-muted-foreground">{s.splitNote}</p>}
    </>
  );

  const rows: TicketRow[] = s.finalSplit.map((row, i) => {
    const delta = s.cashDeltas[i];
    const pos = positionOf.get(row.name);
    const posText = pos ? positionNote(pos, symbol) : "";
    return {
      name: row.name,
      amount: row.amount,
      items: row.items,
      note: (
        <>
          {posText && (
            <span data-testid="position-note" className="block text-xs text-muted-foreground">
              {posText}
            </span>
          )}
          {s.cashRoundingActive && delta !== 0 && (
            <span data-testid="cash-delta" className="block text-xs text-muted-foreground">
              {delta > 0 ? "+" : "−"}{formatMoney(Math.abs(delta), symbol)} rounding
              {i === s.absorberIndex ? " · covers the difference" : ""}
            </span>
          )}
        </>
      ),
    };
  });

  return (
    <div className="space-y-6">
      <ReceiptTicket title={s.billTitle} date={new Date()} symbol={symbol} rows={rows}>
        <SettleUp
          model={{
            shares: s.finalSplit,
            settlement: s.settlement,
            symbol,
            billTitle: s.billTitle,
            tracked: s.settleTracked,
            change: s.changeSettlement,
            stop: s.stopTracking,
            payment,
          }}
        />

        {s.finalSplit.length > 1 && (
          <div className="border-t border-white/5 p-5">
            <AdjustShares s={s} />
          </div>
        )}

        {s.finalSplit.length > 1 && (
          <div className="space-y-3 border-t border-white/5 p-5">
            <div className="text-xs font-semibold text-muted-foreground">Round shares for cash</div>
            <Segmented
              label="Cash rounding"
              value={String(s.cashUnit)}
              onValueChange={(v) => void s.chooseCashRounding(Number(v))}
              options={CASH_UNITS.map((u) => ({ value: String(u), label: u === 1 ? "Exact" : formatMoney(u, symbol) }))}
            />
            {s.cashRoundingActive && (
              <label className="flex min-h-11 items-center justify-between gap-3 text-sm">
                <span className="text-muted-foreground">Covers the difference</span>
                <select
                  data-testid="absorber"
                  value={s.finalSplit[s.absorberIndex]?.name}
                  onChange={(e) => void s.chooseCashRounding(s.cashUnit, e.target.value)}
                  className="h-11 rounded-xl border border-white/10 bg-surface-2 px-3 text-base text-white"
                >
                  {s.finalSplit.map((r) => (
                    <option key={r.name} value={r.name}>{r.name}</option>
                  ))}
                </select>
              </label>
            )}
          </div>
        )}

        <div className="border-t border-white/5">
          <button
            type="button"
            onClick={() => setShowReasoning(!showReasoning)}
            aria-expanded={showReasoning}
            className="flex min-h-12 w-full items-center justify-between px-5 text-xs font-semibold text-muted-foreground transition-colors hover:bg-white/5"
          >
            <span className="flex items-center gap-2"><ScrollText size={14} /> How this was worked out</span>
            {showReasoning ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
          {showReasoning && (
            <div className="px-5 pb-5">
              <div className="max-h-60 overflow-y-auto rounded-xl border border-white/5 bg-black/40 p-4 font-mono text-xs leading-relaxed whitespace-pre-wrap text-zinc-400">
                {s.splitResult}
              </div>
            </div>
          )}
        </div>

        <div className="space-y-3 border-t border-white/5 p-5">
          <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground">
            <Sparkles size={14} className="text-amber-400" /> Change something
          </div>
          <div className="flex gap-2">
            <Input
              data-testid="modify-input"
              aria-label="Change something"
              value={modifyText}
              onChange={(e) => setModifyText(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !s.loading && apply()}
              placeholder={`e.g. ${s.people[0] ?? "Sam"} didn't have the rice`}
              disabled={s.loading}
              enterKeyHint="go"
              className="h-12 border-white/5 bg-surface-2 text-base text-white"
            />
            <Button
              data-testid="modify-apply"
              type="button"
              onClick={apply}
              disabled={s.loading || !modifyText.trim()}
              className="h-12 rounded-xl bg-coral px-5 font-bold text-coral-foreground hover:bg-coral/90"
            >
              {s.loading ? <Loader2 className="size-4 animate-spin" /> : "Apply"}
            </Button>
          </div>
          {notes}
        </div>

        <div className="space-y-3 p-5 pt-0">
          <Button type="button" className="h-12 w-full rounded-xl bg-[#25D366] text-base font-bold text-black hover:bg-[#25D366]/90" onClick={share}>
            <Share2 className="mr-2 size-4" /> Share summary
          </Button>
          <Button type="button" variant="outline" className="h-12 w-full rounded-xl border-white/10 text-sm font-semibold text-zinc-300" onClick={s.finish}>
            Finish session
          </Button>
        </div>
      </ReceiptTicket>
    </div>
  );
}
