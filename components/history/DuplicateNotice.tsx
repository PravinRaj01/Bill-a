"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { Copy } from "lucide-react";
import { toast } from "sonner";

import { hideMyBill } from "@/lib/actions/history";
import { hideShared, type DuplicateView } from "@/lib/actions/shared";

// The same dinner entered twice: you logged it, and a friend shared theirs. Say so, and let the viewer keep whichever
// copy they prefer. A hidden copy leaves History and balances (so nothing is counted twice) and can be brought back.
export function DuplicateNotice({ dup, viewing }: { dup: DuplicateView; viewing: "mine" | "theirs" }) {
  const router = useRouter();
  const [busy, setBusy] = useState(false);

  const hide = async (which: "mine" | "theirs") => {
    setBusy(true);
    try {
      if (which === "mine") await hideMyBill(dup.ownId, true);
      else await hideShared(dup.sharedId, true);
      toast(which === "mine" ? "Hid your copy. Bring it back from History." : `Hid ${dup.ownerName}'s copy. Bring it back from History.`);
      router.push("/dashboard/history", { transitionTypes: ["nav-back"] });
      router.refresh();
    } catch {
      toast.error("Couldn't do that. Please try again.");
      setBusy(false);
    }
  };

  const btn = "min-h-10 rounded-full border border-white/10 px-4 text-xs font-semibold text-foreground transition-colors hover:bg-white/10 disabled:opacity-50";
  return (
    <div data-testid="duplicate-notice" className="space-y-3 rounded-3xl border border-warning/30 bg-warning/10 p-4">
      <div className="flex items-start gap-3">
        <Copy className="mt-0.5 size-4 shrink-0 text-warning" />
        <p className="text-sm">
          {viewing === "mine" ? (
            <>
              <strong>{dup.ownerName}</strong> logged this too (&ldquo;{dup.sharedTitle}&rdquo;). It looks like the same bill, so it may be counted twice.
            </>
          ) : (
            <>You logged this too (&ldquo;{dup.ownTitle}&rdquo;). It looks like the same bill, so it may be counted twice.</>
          )}
        </p>
      </div>
      <div className="flex flex-wrap gap-2">
        <button type="button" data-testid="hide-mine" disabled={busy} onClick={() => void hide("mine")} className={btn}>
          Hide mine
        </button>
        <button type="button" data-testid="hide-theirs" disabled={busy} onClick={() => void hide("theirs")} className={btn}>
          Hide {dup.ownerName}&apos;s
        </button>
      </div>
    </div>
  );
}
