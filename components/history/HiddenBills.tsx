"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";

import { hideMyBill, type HiddenBillEntry } from "@/lib/actions/history";
import { formatMoney } from "@/lib/money";

// Bills you hid because a friend logged the same one. They stay out of History and balances until brought back.
export function HiddenBills({ initial }: { initial: HiddenBillEntry[] }) {
  const [items, setItems] = useState(initial);
  const [open, setOpen] = useState(false);
  const router = useRouter();
  if (items.length === 0) return null;

  const bringBack = async (b: HiddenBillEntry) => {
    try {
      await hideMyBill(b.id, false);
      setItems((all) => all.filter((x) => x.id !== b.id));
      router.refresh();
    } catch {
      toast.error("Couldn't bring that back. Please try again.");
    }
  };

  return (
    <section aria-label="Hidden by you" data-testid="hidden-bills" className="space-y-2">
      <button type="button" aria-expanded={open} onClick={() => setOpen((v) => !v)} className="flex min-h-11 items-center px-1 text-xs font-semibold text-muted-foreground hover:text-foreground">
        {open ? "Hide" : "Show"} {items.length} hidden by you
      </button>
      {open && (
        <ul className="space-y-2">
          {items.map((b) => (
            <li key={b.id} className="flex items-center gap-3 rounded-2xl glass-inset p-3 text-sm">
              <span className="min-w-0 flex-1 truncate text-muted-foreground">
                {b.billTitle} · {formatMoney(b.totalAmount, b.currency)}
              </span>
              <button type="button" data-testid="hidden-bring-back" onClick={() => void bringBack(b)} className="min-h-10 rounded-full px-3 text-xs font-semibold text-foreground hover:bg-white/10">
                Bring back
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
