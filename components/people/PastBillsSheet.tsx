"use client";
import { useEffect, useState } from "react";
import { Check, Loader2 } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Drawer, DrawerContent, DrawerDescription, DrawerHeader, DrawerTitle } from "@/components/ui/drawer";
import { pastBillsFor, sharePastBills, type PastBillView } from "@/lib/actions/friends";
import { formatMoney } from "@/lib/money";
import { cn } from "@/lib/utils";

// After linking a friend: the bills you've already split with them that they can't see yet. Everything is ticked
// to start; untick what you'd rather keep to yourself. Sharing shows them the whole bill (names and amounts).
export function PastBillsSheet({ friend, onClose }: { friend: { id: string; name: string } | null; onClose: () => void }) {
  const [bills, setBills] = useState<PastBillView[] | null>(null);
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!friend) return;
    let cancelled = false;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- reset while the next friend's bills load
    setBills(null);
    pastBillsFor(friend.id)
      .then((b) => {
        if (cancelled) return;
        setBills(b);
        setPicked(new Set(b.map((x) => x.id)));
      })
      .catch(() => !cancelled && setBills([]));
    return () => {
      cancelled = true;
    };
  }, [friend]);

  // Nothing to offer: say so (a sheet that never opens looks like a broken button), then close.
  useEffect(() => {
    if (friend && bills && bills.length === 0) {
      toast(`No past bills with ${friend.name} to share`, { description: "Bills that name them by their name, first name or @username show up here." });
      onClose();
    }
  }, [friend, bills, onClose]);

  const toggle = (id: string) =>
    setPicked((p) => {
      const n = new Set(p);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });

  const share = async () => {
    if (!friend) return;
    setSaving(true);
    try {
      const n = await sharePastBills(friend.id, [...picked]);
      toast.success(n > 0 ? `Shared ${n} ${n === 1 ? "bill" : "bills"} with ${friend.name}` : "Nothing to share");
      onClose();
    } catch {
      toast.error("Couldn't share those. Please try again.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Drawer open={!!friend && !!bills && bills.length > 0} onOpenChange={(o) => !o && onClose()}>
      <DrawerContent data-testid="past-bills">
        <DrawerHeader>
          <DrawerTitle className="text-foreground">Share past bills with {friend?.name}?</DrawerTitle>
          <DrawerDescription className="text-muted-foreground">They&apos;ll see these bills (names and amounts) in their History. Untick any you&apos;d rather keep to yourself, and check that each one is really them.</DrawerDescription>
        </DrawerHeader>
        <div className="mx-auto w-full max-w-md space-y-4 px-4 pb-6">
          <ul className="max-h-[45vh] space-y-1 overflow-y-auto" data-vaul-no-drag>
            {(bills ?? []).map((b) => {
              const on = picked.has(b.id);
              return (
                <li key={b.id}>
                  <button
                    type="button"
                    data-testid="past-bill"
                    aria-pressed={on}
                    onClick={() => toggle(b.id)}
                    className="flex min-h-14 w-full items-center gap-3 rounded-2xl px-3 text-left transition-colors hover:bg-white/5"
                  >
                    <span className={cn("flex size-6 shrink-0 items-center justify-center rounded-md border-2", on ? "border-primary bg-primary" : "border-white/15")}>
                      {on && <Check className="size-4 text-primary-foreground" strokeWidth={4} />}
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-sm font-semibold">{b.billTitle}</span>
                      <span className="block text-xs text-muted-foreground">
                        {new Date(b.createdAt).toISOString().slice(0, 10)}
                        {b.asName.trim().toLowerCase() !== friend?.name.trim().toLowerCase() ? ` · written as "${b.asName}"` : ""}
                      </span>
                    </span>
                    <span className="shrink-0 text-sm font-bold tabular-nums">{formatMoney(b.totalAmount, b.currency)}</span>
                  </button>
                </li>
              );
            })}
          </ul>
          <div className="flex gap-3">
            <Button type="button" data-testid="past-bills-share" className="h-12 flex-1" disabled={saving || picked.size === 0} onClick={share}>
              {saving && <Loader2 className="mr-2 size-4 animate-spin" />} Share {picked.size > 0 ? picked.size : ""}
            </Button>
            <Button type="button" variant="glass" className="h-12 px-5" onClick={onClose}>
              Not now
            </Button>
          </div>
        </div>
      </DrawerContent>
    </Drawer>
  );
}
