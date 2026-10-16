"use client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { SettleModel } from "@/components/split/SettleUp";
import type { BillHistoryRow } from "@/lib/db/schema";
import { payloadWithSettlement } from "@/lib/sync/billPayload";
import { enqueueBill } from "@/lib/sync/outbox";
import { defaultSettlement, guessMe, localDate, normalizeSettlement, settleView } from "@/lib/split/settle";
import type { Settlement } from "@/types/domain";

// Settle-up for a bill that is already saved (the History detail page). Same rules as the Summary: what
// is shown is the saved settlement or, until the user touches it, a default; the first change is what
// starts tracking. Edits go through the local-first outbox with the bill's own clientId, so the server
// row is UPDATED (never duplicated) and a change made offline syncs later.

export function useBillSettlement(bill: BillHistoryRow | null, user: { id: string; name: string | null } | null): SettleModel | null {
  // The user's edits for THIS bill; until there are any, the saved settlement on the loaded row is shown.
  const [edited, setEdited] = useState<{ billId: string; value: Settlement | null } | null>(null);
  const saved = bill ? (edited && edited.billId === bill.id ? edited.value : (bill.data.settlement ?? null)) : null;

  const shares = useMemo(() => bill?.data.split ?? [], [bill]);
  const settlement = useMemo(() => {
    if (!bill || shares.length < 2) return null;
    const me = guessMe(shares.map((s) => s.name), user?.name);
    return normalizeSettlement(saved ?? defaultSettlement(shares, me), shares);
  }, [bill, shares, saved, user?.name]);

  const save = async (next: Settlement | null) => {
    if (!bill || !user) return;
    await enqueueBill(user.id, payloadWithSettlement(bill, next));
  };

  const changeNow = async (edit: (s: Settlement) => Settlement) => {
    if (!bill || !settlement) return;
    let next = edit(settlement);
    const view = settleView(shares, next, localDate());
    if (view && view.ok) next = view.settlement; // drop paid marks that no longer match
    setEdited({ billId: bill.id, value: next });
    await save(next);
  };

  // A stable function that always runs the latest version, so an Undo toast made earlier can't act on
  // (and overwrite with) a stale copy of the settlement.
  const ref = useRef(changeNow);
  useEffect(() => {
    ref.current = changeNow;
  });
  const change = useCallback((edit: (s: Settlement) => Settlement) => ref.current(edit), []);

  if (!bill) return null;
  return {
    shares,
    settlement,
    symbol: bill.currency || "RM",
    billTitle: bill.billTitle,
    tracked: saved !== null,
    change,
    stop: async () => {
      setEdited({ billId: bill.id, value: null });
      await save(null);
    },
  };
}
