"use client";
import { Calculator, Camera, ChevronRight } from "lucide-react";

import type { SplitSession } from "@/hooks/use-split-session";

const tile =
  "flex min-h-24 items-center gap-4 rounded-3xl glass p-5 text-left outline-none transition-colors hover:border-white/25 focus-visible:ring-2 focus-visible:ring-white/30 active:scale-[0.98]";

// Who first, then how: the people are set, now where does the total come from? A scanned receipt (the
// items are read and split between people) or just a typed amount (split evenly or by hand).
export function HowStep({ s }: { s: SplitSession }) {
  return (
    <div className="space-y-6">
      <div className="space-y-1">
        <h2 className="type-title">How are you splitting?</h2>
        <p className="text-sm text-muted-foreground">Step 2 of 3 · {s.people.length} {s.people.length === 1 ? "person" : "people"} in</p>
      </div>

      <div className="space-y-3">
        <button type="button" data-testid="how-scan" className={tile} onClick={() => s.chooseHow("scan")}>
          <span className="flex size-12 shrink-0 items-center justify-center rounded-full bg-coral/15 text-coral">
            <Camera className="size-6" />
          </span>
          <span className="min-w-0 flex-1">
            <span className="block text-base font-bold">Scan a receipt</span>
            <span className="block text-sm text-muted-foreground">We read the items, you say who had what</span>
          </span>
          <ChevronRight size={18} className="shrink-0 text-muted-foreground" />
        </button>
        <button type="button" data-testid="how-amount" className={tile} onClick={() => s.chooseHow("amount")}>
          <span className="flex size-12 shrink-0 items-center justify-center rounded-full bg-white/10">
            <Calculator className="size-6" />
          </span>
          <span className="min-w-0 flex-1">
            <span className="block text-base font-bold">Type an amount</span>
            <span className="block text-sm text-muted-foreground">No receipt, just a total to share out</span>
          </span>
          <ChevronRight size={18} className="shrink-0 text-muted-foreground" />
        </button>
      </div>
    </div>
  );
}
