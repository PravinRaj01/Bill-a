"use client";

import { Scale } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { SplitResult } from "@/types/domain";

// Shown when the AI's split and the on-device rules' split disagree. A wrong AI answer looks
// exactly like a right one, so instead of silently trusting either, both are laid out with
// the people whose amounts differ highlighted, and the user picks the one they meant.

export type Reading = "ai" | "rules";

function Column({
  title,
  subtitle,
  result,
  other,
  symbol,
  onPick,
  testId,
}: {
  title: string;
  subtitle: string;
  result: SplitResult;
  other: SplitResult;
  symbol: string;
  onPick: () => void;
  testId: string;
}) {
  return (
    <div data-testid={testId} className="flex flex-1 flex-col gap-3 rounded-2xl border border-white/10 bg-black/40 p-4">
      <div>
        <p className="text-xs font-bold text-white">{title}</p>
        <p className="text-[10px] uppercase tracking-widest text-zinc-600">{subtitle}</p>
      </div>
      <ul className="space-y-2">
        {result.splits.map((s, i) => {
          const differs = s.amount !== other.splits[i]?.amount;
          return (
            <li key={s.name} className={`rounded-lg px-2 py-1.5 ${differs ? "bg-amber-500/10 ring-1 ring-amber-500/30" : ""}`}>
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-sm font-bold text-white">{s.name}</span>
                <span className="font-mono text-sm text-white">{symbol}{(s.amount / 100).toFixed(2)}</span>
              </div>
              <p className="mt-0.5 text-[10px] leading-snug text-zinc-500">{s.items}</p>
            </li>
          );
        })}
      </ul>
      <Button type="button" onClick={onPick} className="mt-auto h-9 rounded-full bg-white text-[11px] font-black uppercase tracking-widest text-black hover:bg-zinc-200">
        Use this reading
      </Button>
    </div>
  );
}

export function CrossCheck({
  aiName,
  ai,
  rules,
  symbol,
  onPick,
}: {
  aiName: string;
  ai: SplitResult;
  rules: SplitResult;
  symbol: string;
  onPick: (which: Reading) => void;
}) {
  return (
    <div data-testid="crosscheck" className="space-y-3 rounded-2xl border border-amber-500/30 bg-amber-500/5 p-4">
      <div className="flex items-start gap-2 text-amber-200">
        <Scale className="mt-0.5 h-4 w-4 shrink-0" />
        <div className="space-y-0.5">
          <p className="text-xs font-bold">Two readings of your instruction gave different splits</p>
          <p className="text-[11px] text-amber-200/70">
            {aiName} and Bill.a&apos;s built-in rules understood it differently. The highlighted amounts differ — pick the one you meant.
          </p>
        </div>
      </div>
      <div className="flex flex-col gap-3 sm:flex-row">
        <Column testId="crosscheck-ai" title={`${aiName}'s reading`} subtitle="AI" result={ai} other={rules} symbol={symbol} onPick={() => onPick("ai")} />
        <Column testId="crosscheck-rules" title="Rule-based reading" subtitle="Built-in rules, on your device" result={rules} other={ai} symbol={symbol} onPick={() => onPick("rules")} />
      </div>
    </div>
  );
}
