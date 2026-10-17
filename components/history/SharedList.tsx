"use client";
import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { toast } from "sonner";

import { PersonAvatar } from "@/components/person-avatar";
import { hideShared, type SharedListEntry } from "@/lib/actions/shared";
import { formatMoney } from "@/lib/money";
import { cn } from "@/lib/utils";

// Bills friends have shared with you, under your own History: who shared it, and what runs between you.
// Read-only (the friend owns them). Hidden ones sit behind a link so they can be brought back.
export function SharedList({ initial }: { initial: SharedListEntry[] }) {
  const [entries, setEntries] = useState(initial);
  const [showHidden, setShowHidden] = useState(false);
  const router = useRouter();
  const visible = entries.filter((e) => !e.hidden);
  const hidden = entries.filter((e) => e.hidden);
  if (entries.length === 0) return null;

  const unhide = async (e: SharedListEntry) => {
    try {
      await hideShared(e.id, false);
      setEntries((all) => all.map((x) => (x.id === e.id ? { ...x, hidden: false } : x)));
      router.refresh();
    } catch {
      toast.error("Couldn't bring that back. Please try again.");
    }
  };

  return (
    <section aria-label="Shared with you" data-testid="shared-list" className="space-y-3">
      <h2 className="type-eyebrow px-1">Shared with you</h2>
      <ul className="space-y-3">
        {visible.map((e) => (
          <li key={e.id}>
            <Link
              href={`/dashboard/shared/${e.id}`}
              transitionTypes={["nav-forward"]}
              data-testid="shared-row"
              className="flex items-center gap-4 rounded-2xl glass p-5 transition-colors hover:border-white/20"
            >
              <PersonAvatar name={e.owner.name} seed={e.owner.avatarSeed} className="size-10" />
              <span className="min-w-0 flex-1">
                <span className="block truncate text-base font-bold">{e.billTitle}</span>
                <span className="block truncate text-xs text-muted-foreground">Shared by {e.owner.name} · {e.createdAt.slice(0, 10)}</span>
              </span>
              <span className="shrink-0 text-right">
                <span className="block text-base font-bold tabular-nums">{formatMoney(e.totalAmount, e.currency || "RM")}</span>
                {e.mine && (
                  <span data-testid="shared-status" className={cn("block text-xs font-semibold", e.mine.settled ? "text-success" : e.mine.iOwe > 0 ? "text-coral" : "text-muted-foreground")}>
                    {e.mine.settled ? "Settled" : e.mine.iOwe > 0 ? `You owe ${formatMoney(e.mine.iOwe, e.currency || "RM")}` : e.mine.theyOwe > 0 ? `Owes you ${formatMoney(e.mine.theyOwe, e.currency || "RM")}` : ""}
                  </span>
                )}
              </span>
            </Link>
          </li>
        ))}
      </ul>
      {hidden.length > 0 && (
        <div className="space-y-2">
          <button type="button" data-testid="shared-hidden-toggle" aria-expanded={showHidden} onClick={() => setShowHidden((v) => !v)} className="flex min-h-11 items-center px-1 text-xs font-semibold text-muted-foreground hover:text-foreground">
            {showHidden ? "Hide" : "Show"} {hidden.length} hidden
          </button>
          {showHidden && (
            <ul className="space-y-2">
              {hidden.map((e) => (
                <li key={e.id} className="flex items-center gap-3 rounded-2xl glass-inset p-3 text-sm">
                  <span className="min-w-0 flex-1 truncate text-muted-foreground">{e.billTitle} · {e.owner.name}</span>
                  <button type="button" data-testid="shared-unhide" onClick={() => unhide(e)} className="min-h-10 rounded-full px-3 text-xs font-semibold text-foreground hover:bg-white/10">
                    Bring back
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}
