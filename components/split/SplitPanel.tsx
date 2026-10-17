"use client";
import { useState } from "react";
import { Loader2 } from "lucide-react";

import { ClarifyChips } from "@/components/ai/ClarifyChips";
import { CrossCheck } from "@/components/ai/CrossCheck";
import { SharesEditor } from "@/components/split/SharesEditor";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Segmented } from "@/components/ui/segmented";
import { useShareEditor } from "@/hooks/use-share-editor";
import type { SplitSession } from "@/hooks/use-split-session";
import { formatMoney } from "@/lib/money";

export type Mode = "equal" | "custom" | "describe";

const MODE_LABELS: Record<Mode, string> = { equal: "Equal", custom: "Custom", describe: "Describe" };

/** Tappable examples under the instruction box, using the actual people (so they work as typed). */
function exampleInstructions(people: string[]): string[] {
  const out = ["Split equally"];
  if (people.length >= 2) out.push(`${people[0]} 70%, ${people[1]} 30%`);
  if (people.length >= 3) out.push(`${people[0]} pays RM20, the rest equally`);
  return out;
}

// "How to split": three ways to get from the items to who pays what.
//   Equal    everyone the same. One tap, works offline.
//   Custom   a slider per person, exact to the cent. Works offline.
//   Describe say it in words ("Alexis pays the drinks"): the on-device rules, or an AI if you set a key.
// Key this component by the people list so a different group starts fresh (equal, unlocked).
export function SplitPanel({ s, modes = ["equal", "custom", "describe"] }: { s: SplitSession; modes?: Mode[] }) {
  // Describe is the default where it is offered (the scan flow); otherwise the first mode (Equal: one tap).
  const [mode, setMode] = useState<Mode>(modes.includes("describe") ? "describe" : modes[0]);
  const custom = useShareEditor(s.people, s.displayedTotal);
  const equal = useShareEditor(s.people, s.displayedTotal); // never edited: always the equal preview

  const ready = !s.loading && !!s.receipt && s.receipt.items.length > 0 && s.displayedTotal > 0;
  const go = () => {
    if (mode === "describe") void s.split();
    else void s.applyShares(mode === "equal" ? equal.amounts : custom.amounts);
  };

  return (
    <section aria-label="How to split" className="rounded-3xl glass p-5">
      <div className="mb-4 flex items-baseline justify-between gap-3">
        <h3 className="text-lg font-bold text-foreground">How to split</h3>
        <span className="text-sm text-muted-foreground">
          {formatMoney(s.displayedTotal, s.symbol)}
          {s.includeTax ? " incl. tax" : ""}
        </span>
      </div>

      <Segmented
        label="How to split"
        value={mode}
        onValueChange={setMode}
        options={modes.map((m) => ({ value: m, label: MODE_LABELS[m] }))}
      />

      <div className="mt-4 space-y-4">
        {mode === "equal" && <SharesEditor people={s.people} symbol={s.symbol} editor={equal} readOnly />}
        {mode === "custom" && <SharesEditor people={s.people} symbol={s.symbol} editor={custom} />}
        {mode === "describe" && (
          <>
            <Input
              aria-label="How to split"
              placeholder={`How to split (e.g. ${s.people[0] ?? "Sam"} pays the drinks)`}
              value={s.instruction}
              onChange={(e) => s.setInstruction(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !s.loading && s.split()}
              enterKeyHint="go"
             
            />
            <div className="flex flex-wrap gap-2" aria-label="Examples">
              {exampleInstructions(s.people).map((ex) => (
                <button
                  key={ex}
                  type="button"
                  onClick={() => s.setInstruction(ex)}
                  className="min-h-9 rounded-full border border-white/10 bg-white/5 px-3 text-xs font-medium text-foreground/80 transition-colors hover:bg-white/10 hover:text-foreground"
                >
                  {ex}
                </button>
              ))}
            </div>
            {s.clarify && <ClarifyChips chips={s.clarify.chips} hasPreview={!!s.clarify.preview} onAction={s.handleClarify} />}
            {s.disagree && <CrossCheck aiName={s.disagree.tier === "groq" ? "Groq" : "Gemini"} ai={s.disagree.ai.result} rules={s.disagree.rules.result} symbol={s.symbol} onPick={s.pickReading} />}
            {s.splitNote && !s.clarify && <p data-testid="split-note" className="px-1 text-xs text-muted-foreground">{s.splitNote}</p>}
          </>
        )}
      </div>

      {/* Phones: pinned above the bottom nav. Tablet portrait (no bottom nav): pinned to the bottom edge. lg+: sits in the card. */}
      <div className="sticky bottom-[calc(4.75rem+env(safe-area-inset-bottom))] z-10 -mx-5 mt-5 bg-gradient-to-t from-surface via-surface to-transparent px-5 pt-3 pb-1 md:bottom-4 lg:static lg:mx-0 lg:bg-none lg:p-0">
        <Button
          type="button"
          data-testid="split-button"
          className="h-12 w-full"
          onClick={go}
          disabled={!ready}
        >
          {s.loading ? <Loader2 className="animate-spin" /> : mode === "describe" ? "Split bill" : `Split ${formatMoney(s.displayedTotal, s.symbol)}`}
        </Button>
      </div>
    </section>
  );
}
