"use client";
import { useState, type ReactNode } from "react";
import { ChevronDown, ChevronUp, ImageIcon, Loader2, ScrollText, Share2, Sparkles } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ClarifyChips } from "@/components/ai/ClarifyChips";
import { CrossCheck } from "@/components/ai/CrossCheck";
import { ReceiptTicket, type TicketRow } from "@/components/receipt/ReceiptTicket";
import { ShareSheet } from "@/components/receipt/ShareSheet";
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

type Tab = "settle" | "adjust" | "share";

// The receipt ticket stays on top; what you can do with it sits behind three tabs so a phone isn't a
// two-screen scroll. Prompts the split itself needs answered (a question from the reader, two readings that
// disagree) stay above the tabs, where they can't be missed. With one person there is nobody to settle with
// or adjust between, so the tabs give way to just the share buttons.
export function SummaryStep({ s }: { s: SplitSession }) {
  const [showReasoning, setShowReasoning] = useState(false);
  const [modifyText, setModifyText] = useState("");
  const { symbol } = s;
  const payment = usePaymentShare();
  const [imageOpen, setImageOpen] = useState(false);
  const several = s.finalSplit.length > 1;
  // Settling up is what most people do next, so it opens first.
  const [tab, setTab] = useState<Tab>("settle");
  const active: Tab = several ? tab : "share";

  const apply = () => {
    const text = modifyText.trim();
    if (!text) return;
    setModifyText("");
    s.modify(text);
  };

  // Settle-up state of this bill (null until there are two people to settle between).
  const settleNow = s.settlement && several ? settleView(s.finalSplit, s.settlement, localDate()) : null;
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

  const settleTab: ReactNode = (
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
        participants: s.participants,
      }}
    />
  );

  const adjustTab: ReactNode = (
    <>
      {several && (
        <div className="border-t border-white/8 p-5">
          <AdjustShares s={s} />
        </div>
      )}

      {several && (
        <div className="space-y-3 border-t border-white/8 p-5">
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
                className="h-11 glass-inset rounded-xl px-3 text-base text-foreground"
              >
                {s.finalSplit.map((r) => (
                  <option key={r.name} value={r.name}>{r.name}</option>
                ))}
              </select>
            </label>
          )}
        </div>
      )}

      <div className="space-y-3 border-t border-white/8 p-5">
        <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground">
          <Sparkles size={14} className="text-warning" /> Change something
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
            className="h-12 border-white/8 bg-white/10 text-base text-foreground"
          />
          <Button
            data-testid="modify-apply"
            type="button"
            onClick={apply}
            disabled={s.loading || !modifyText.trim()}
            className="h-12 px-5"
          >
            {s.loading ? <Loader2 className="size-4 animate-spin" /> : "Apply"}
          </Button>
        </div>
        {s.splitNote && !s.clarify && <p data-testid="split-note" className="text-xs text-muted-foreground">{s.splitNote}</p>}
      </div>

      <div className="border-t border-white/8">
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
            <div className="max-h-60 overflow-y-auto rounded-xl border border-white/8 glass-inset p-4 font-mono text-xs leading-relaxed whitespace-pre-wrap text-muted-foreground">
              {s.splitResult}
            </div>
          </div>
        )}
      </div>
    </>
  );

  const shareTab: ReactNode = (
    <div className="space-y-3 border-t border-white/8 p-5">
      <Button type="button" variant="whatsapp" size="lg" data-testid="share-summary" className="w-full" onClick={share}>
        <Share2 className="mr-2 size-4" /> Share summary
      </Button>
      <Button
        type="button"
        variant="outline"
        data-testid="share-image-open"
        className="h-12 w-full rounded-xl border-white/10 text-base font-semibold text-foreground"
        onClick={() => setImageOpen(true)}
      >
        <ImageIcon className="mr-2 size-4" /> Share as image
      </Button>
      <Button type="button" variant="outline" className="h-12 w-full rounded-xl border-white/10 text-sm font-semibold text-foreground/80" onClick={s.finish}>
        Finish session
      </Button>
    </div>
  );

  return (
    <div className="space-y-6">
      <ReceiptTicket title={s.billTitle} date={new Date()} symbol={symbol} rows={rows}>
        {(s.clarify || s.disagree) && (
          <div className="space-y-3 border-t border-white/8 p-5">
            {s.clarify && <ClarifyChips chips={s.clarify.chips} hasPreview={!!s.clarify.preview} onAction={s.handleClarify} />}
            {s.disagree && <CrossCheck aiName={s.disagree.tier === "groq" ? "Groq" : "Gemini"} ai={s.disagree.ai.result} rules={s.disagree.rules.result} symbol={symbol} onPick={s.pickReading} />}
          </div>
        )}

        {several ? (
          <div className="border-t border-white/8 px-5 pt-4 pb-1">
            <Segmented
              label="Summary sections"
              value={tab}
              onValueChange={setTab}
              options={[
                { value: "settle", label: "Settle up" },
                { value: "adjust", label: "Adjust" },
                { value: "share", label: "Share" },
              ]}
            />
          </div>
        ) : (
          // One person: no tabs, so "Change something" (the only adjusting left) sits above the share buttons.
          adjustTab
        )}

        <div data-testid={`summary-tab-${active}`}>
          {active === "settle" && settleTab}
          {active === "adjust" && adjustTab}
          {active === "share" && shareTab}
        </div>
      </ReceiptTicket>
      <ShareSheet
        open={imageOpen}
        onOpenChange={setImageOpen}
        data={{ title: s.billTitle, date: localDate(), symbol, shares: s.finalSplit, settlement: s.settlement, view: settleNow }}
        payment={payment}
      />
    </div>
  );
}
