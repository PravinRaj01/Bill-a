"use client";
import { Check, X } from "lucide-react";

import { PersonAvatar } from "@/components/person-avatar";
import type { PendingClaim } from "@/lib/actions/shared";
import { formatMoney } from "@/lib/money";

// On your own bill: friends who say they've paid you. Confirm marks that payment paid (the same mark you'd
// make by hand); "Not received" tells them it hasn't arrived, and they can say so again later.
export function ClaimsBanner({
  claims,
  symbol,
  onConfirm,
  onReject,
}: {
  claims: PendingClaim[];
  symbol: string;
  onConfirm: (c: PendingClaim) => void | Promise<void>;
  onReject: (c: PendingClaim) => void | Promise<void>;
}) {
  if (claims.length === 0) return null;
  return (
    <ul data-testid="claims" className="space-y-2">
      {claims.map((c) => (
        <li key={c.id} data-testid="claim" className="flex items-center gap-3 rounded-3xl border border-success/30 bg-success/10 p-3">
          <PersonAvatar name={c.from.name} seed={c.from.avatarSeed} className="size-10" />
          <span className="min-w-0 flex-1">
            <span className="block text-sm font-bold">{c.from.name} says they paid you</span>
            <span className="block text-xs text-muted-foreground tabular-nums">{formatMoney(c.amount, symbol)}</span>
          </span>
          <button type="button" data-testid="claim-confirm" onClick={() => onConfirm(c)} className="flex min-h-10 items-center gap-1 rounded-full bg-success px-4 text-sm font-bold text-background">
            <Check size={14} strokeWidth={3} /> Got it
          </button>
          <button type="button" aria-label={`Not received from ${c.from.name}`} data-testid="claim-reject" onClick={() => onReject(c)} className="flex size-10 items-center justify-center rounded-full text-muted-foreground hover:bg-white/10">
            <X size={16} />
          </button>
        </li>
      ))}
    </ul>
  );
}
