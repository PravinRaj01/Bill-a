"use client";
import { useState } from "react";
import { ChevronDown, ChevronUp, Loader2, ScrollText, Share2, Sparkles } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ClarifyChips } from "@/components/ai/ClarifyChips";
import { CrossCheck } from "@/components/ai/CrossCheck";
import { PersonAvatar } from "@/components/person-avatar";
import { Money } from "@/components/ui/money";
import { formatMoney } from "@/lib/money";
import type { SplitSession } from "@/hooks/use-split-session";

export function SummaryStep({ s }: { s: SplitSession }) {
  const [showReasoning, setShowReasoning] = useState(false);
  const [modifyText, setModifyText] = useState("");
  const { symbol } = s;

  const apply = () => {
    const text = modifyText.trim();
    if (!text) return;
    setModifyText("");
    s.modify(text);
  };

  const share = () => {
    let text = `*Bill-a Summary (${symbol})*\n\n`;
    s.structuredSplit.forEach((r) => (text += `👤 *${r.name}*: ${formatMoney(r.amount, symbol)}\n`));
    window.open(`whatsapp://send?text=${encodeURIComponent(text)}`);
  };

  const notes = (
    <>
      {s.clarify && <ClarifyChips chips={s.clarify.chips} hasPreview={!!s.clarify.preview} onAction={s.handleClarify} />}
      {s.disagree && <CrossCheck aiName={s.disagree.tier === "groq" ? "Groq" : "Gemini"} ai={s.disagree.ai.result} rules={s.disagree.rules.result} symbol={symbol} onPick={s.pickReading} />}
      {s.splitNote && !s.clarify && <p data-testid="split-note" className="text-xs text-muted-foreground">{s.splitNote}</p>}
    </>
  );

  return (
    <div className="space-y-6">
      <div className="overflow-hidden rounded-3xl border border-white/5 bg-surface shadow-2xl ring-1 ring-white/10">
        <div className="border-b border-white/5 bg-white/[0.02] py-4 text-center text-xs font-semibold tracking-widest text-muted-foreground uppercase">
          Final settlement
        </div>

        <ul className="divide-y divide-white/5">
          {s.structuredSplit.map((row) => (
            <li key={row.name} className="flex items-center gap-3 px-6 py-4">
              <PersonAvatar name={row.name} />
              <span className="flex-1 truncate text-base font-bold text-white">{row.name}</span>
              <Money cents={row.amount} currency={symbol} className="text-xl font-bold text-white" />
            </li>
          ))}
        </ul>

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
              placeholder="e.g. Aisha didn't have the rice"
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
            <Share2 className="mr-2 size-4" /> Share via WhatsApp
          </Button>
          <Button type="button" variant="outline" className="h-12 w-full rounded-xl border-white/10 text-sm font-semibold text-zinc-300" onClick={s.finish}>
            Finish session
          </Button>
        </div>
      </div>
    </div>
  );
}
