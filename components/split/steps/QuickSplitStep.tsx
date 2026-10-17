"use client";
import { SplitPanel } from "@/components/split/SplitPanel";
import type { SplitSession } from "@/hooks/use-split-session";
import { formatMoney } from "@/lib/money";

// Quick split, step 3: Equal (one tap) or Custom (sliders). There are no items for an instruction to
// refer to, so Describe isn't offered; the "describe it" follow-up on the result page still accepts
// "Alexis 70%, Riley 30%".
export function QuickSplitStep({ s }: { s: SplitSession }) {
  const item = s.receipt?.items[0]?.name;
  return (
    <div className="space-y-6">
      <div className="space-y-1">
        <h2 className="type-title">How should it be split?</h2>
        <p className="text-sm text-muted-foreground">
          Step 3 of 3 · {formatMoney(s.displayedTotal, s.symbol)}
          {item && item !== "Quick split" ? ` · ${item}` : ""}
        </p>
      </div>
      <SplitPanel key={s.people.join("|")} s={s} modes={["equal", "custom"]} />
    </div>
  );
}
