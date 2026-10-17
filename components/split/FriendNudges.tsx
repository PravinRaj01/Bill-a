"use client";
import { useEffect, useState } from "react";
import { BellRing } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { nudgeFriend, nudgeTargets, type NudgeTarget } from "@/lib/actions/notifications";
import { formatMoney } from "@/lib/money";

// On your own bill: friends on Bill.a who still owe you get a reminder in their inbox, once a day. The amount
// and the wording come from the server, so there is nothing to type or to get wrong.
export function FriendNudges({ billId, symbol }: { billId: string; symbol: string }) {
  const [targets, setTargets] = useState<NudgeTarget[]>([]);
  const [busy, setBusy] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    nudgeTargets(billId).then((t) => !cancelled && setTargets(t)).catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [billId]);

  if (targets.length === 0) return null;

  const send = async (t: NudgeTarget) => {
    setBusy(t.friendId);
    try {
      const r = await nudgeFriend(billId, t.friendId);
      if (r.ok) {
        toast(`Reminded ${t.name}`);
        setTargets((all) => all.map((x) => (x.friendId === t.friendId ? { ...x, nudgedToday: true } : x)));
      } else {
        toast(r.error);
        if (r.error.includes("already")) setTargets((all) => all.map((x) => (x.friendId === t.friendId ? { ...x, nudgedToday: true } : x)));
      }
    } catch {
      toast.error("Couldn't send that. Please try again.");
    } finally {
      setBusy(null);
    }
  };

  return (
    <ul data-testid="friend-nudges" className="space-y-2">
      {targets.map((t) => (
        <li key={t.friendId} className="flex items-center gap-3 rounded-3xl glass p-3 pl-4">
          <span className="min-w-0 flex-1">
            <span className="block truncate text-sm font-bold">{t.name} owes you</span>
            <span className="block text-xs text-muted-foreground tabular-nums">{formatMoney(t.owed, symbol)}</span>
          </span>
          <Button type="button" variant="glass" size="sm" data-testid="friend-nudge" disabled={t.nudgedToday || busy === t.friendId} onClick={() => void send(t)}>
            <BellRing /> {t.nudgedToday ? "Reminded today" : "Remind in app"}
          </Button>
        </li>
      ))}
    </ul>
  );
}
