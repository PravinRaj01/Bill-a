"use client";
import { useState } from "react";
import { SlidersHorizontal } from "lucide-react";

import { SharesEditor } from "@/components/split/SharesEditor";
import { Button } from "@/components/ui/button";
import { useShareEditor } from "@/hooks/use-share-editor";
import type { SplitSession } from "@/hooks/use-split-session";

// "Adjust shares" on the final page: the sliders, seeded from what was just worked out (the
// exact shares, before any cash rounding). Applying replaces the split and re-saves the bill.
function AdjustPanel({ s, onClose }: { s: SplitSession; onClose: () => void }) {
  const names = s.structuredSplit.map((r) => r.name);
  const seed = s.structuredSplit.map((r) => r.amount);
  const editor = useShareEditor(names, seed.reduce((a, b) => a + b, 0), seed);

  return (
    <div data-testid="adjust-panel" className="space-y-4">
      <SharesEditor people={names} symbol={s.symbol} editor={editor} />
      <div className="flex gap-2">
        <Button type="button" variant="outline" className="h-12 flex-1 rounded-xl border-white/10 text-sm font-semibold text-foreground/80" onClick={onClose}>
          Cancel
        </Button>
        <Button
          type="button"
          data-testid="adjust-apply"
          className="h-12 flex-1"
          onClick={async () => {
            await s.applyShares(editor.amounts);
            onClose();
          }}
        >
          Apply shares
        </Button>
      </div>
    </div>
  );
}

export function AdjustShares({ s }: { s: SplitSession }) {
  const [open, setOpen] = useState(false);
  return open ? (
    <AdjustPanel s={s} onClose={() => setOpen(false)} />
  ) : (
    <Button
      type="button"
      variant="outline"
      data-testid="adjust-open"
      className="h-12 w-full rounded-xl border-white/10 text-sm font-semibold text-foreground"
      onClick={() => setOpen(true)}
    >
      <SlidersHorizontal className="mr-2 size-4" /> Adjust shares
    </Button>
  );
}
