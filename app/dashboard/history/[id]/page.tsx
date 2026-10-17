"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { ChevronDown, ChevronLeft, ChevronUp, ImageIcon, Loader2, Play, ScrollText, Share2, Trash2 } from "lucide-react";
import { toast } from "sonner";

import { PageTransition } from "@/components/page-transition";
import { ReceiptTicket, type TicketRow } from "@/components/receipt/ReceiptTicket";
import { ShareSheet } from "@/components/receipt/ShareSheet";
import { ClaimsBanner } from "@/components/split/ClaimsBanner";
import { FriendNudges } from "@/components/split/FriendNudges";
import { SettleUp } from "@/components/split/SettleUp";
import { ConfirmSheet } from "@/components/ui/confirm-sheet";
import { useBillSettlement } from "@/hooks/use-bill-settlement";
import { usePaymentShare } from "@/hooks/use-payment-share";
import { deleteBills, getBill } from "@/lib/actions/history";
import { pendingClaims, rejectPaidClaim, type PendingClaim } from "@/lib/actions/shared";
import { getCurrentUser } from "@/lib/actions/user";
import type { BillHistoryRow } from "@/lib/db/schema";
import { allOwedToMe, localDate, markPaid, positionNote, positions, settleView, summaryText } from "@/lib/split/settle";
import { shareMessage, shareText } from "@/lib/ui/share";
import { Button } from "@/components/ui/button";

export default function BillDetailPage() {
  const params = useParams();
  const id = params?.id as string;

  const [bill, setBill] = useState<BillHistoryRow | null>(null);
  const [loading, setLoading] = useState(true);
  const [user, setUser] = useState<{ id: string; name: string | null } | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [showReasoning, setShowReasoning] = useState(false);
  const [imageOpen, setImageOpen] = useState(false);
  const [claims, setClaims] = useState<PendingClaim[]>([]);
  const router = useRouter();
  const payment = usePaymentShare();

  // Who is signed in (edits to settle-up are saved as theirs, through the outbox).
  useEffect(() => {
    getCurrentUser().then((u) => setUser(u ? { id: u.id, name: u.name } : null)).catch(() => setUser(null));
  }, []);
  const settle = useBillSettlement(bill, user);

  useEffect(() => {
    if (!id) return;
    const fetchBill = async () => {
      // Scoped to the signed-in user on the server: an id that belongs to someone else is
      // indistinguishable from one that doesn't exist.
      const data = await getBill(id).catch(() => null);
      if (!data) router.push("/dashboard/history");
      else setBill(data);
      setLoading(false);
    };
    fetchBill();
  }, [id, router]);

  // Friends who say they've paid you for this bill (only bills shared with friends can have any).
  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    pendingClaims(id).then((c) => !cancelled && setClaims(c)).catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [id]);

  const confirmClaim = async (c: PendingClaim) => {
    // The same paid mark you'd make by hand; saving it is what confirms the friend's claim on the server.
    await settle?.change((s) => markPaid(s, { key: c.transferKey, amount: c.amount }, localDate()));
    setClaims((all) => all.filter((x) => x.id !== c.id));
    toast(`Marked paid by ${c.from.name}`);
  };
  const rejectClaim = async (c: PendingClaim) => {
    try {
      await rejectPaidClaim(c.id);
      setClaims((all) => all.filter((x) => x.id !== c.id));
      toast(`Told ${c.from.name} it hasn't arrived`);
    } catch {
      toast.error("Couldn't do that. Please try again.");
    }
  };

  // Back slides to where the bill was opened from (Balances links here with ?from=balances), else History.
  const goBack = () => {
    const from = new URLSearchParams(window.location.search).get("from");
    router.push(from === "balances" ? "/dashboard/balances" : "/dashboard/history", { transitionTypes: ["nav-back"] });
  };

  const handleDelete = async () => {
    await deleteBills([id]);
    router.push("/dashboard/history", { transitionTypes: ["nav-back"] });
    router.refresh();
  };

  // The new-session page loads the bill itself (scoped to the signed-in user on the server), so no data
  // is passed through sessionStorage.
  const handleContinue = () => {
    router.push(`/dashboard/new?restore=${encodeURIComponent(id)}`, { transitionTypes: ["nav-forward"] });
  };

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-background text-foreground">
        <Loader2 className="animate-spin" />
      </div>
    );
  }
  if (!bill) return null;

  const shares = bill.data.split;
  const symbol = bill.currency || "RM";
  const view = settle?.settlement ? settleView(shares, settle.settlement, localDate()) : null;
  const positionOf = new Map(
    view?.ok && settle?.settlement ? positions(shares, settle.settlement.paidBy).map((p) => [p.name, p] as const) : [],
  );

  const rows: TicketRow[] = shares.map((row) => {
    const pos = positionOf.get(row.name);
    const note = pos ? positionNote(pos, symbol) : "";
    return {
      name: row.name,
      amount: row.amount,
      items: row.items,
      note: note ? (
        <span data-testid="position-note" className="block text-xs text-muted-foreground">
          {note}
        </span>
      ) : undefined,
    };
  });

  const share = async () => {
    const text = summaryText({
      currency: symbol,
      shares,
      settlement: settle?.settlement,
      view,
      payment: allOwedToMe(view, settle?.settlement?.me) ? payment.block : undefined,
    });
    const msg = shareMessage(await shareText(text));
    if (msg) toast(msg);
  };

  return (
    <PageTransition>
      <div className="mx-auto mb-20 max-w-2xl space-y-6 p-6 text-foreground md:p-10">
        <div className="flex items-center justify-between">
          <button
            type="button"
            onClick={goBack}
            className="flex min-h-11 items-center pr-3 text-sm font-semibold text-muted-foreground transition-colors hover:text-foreground"
          >
            <ChevronLeft className="mr-1 size-4" /> Back
          </button>
          <button
            type="button"
            aria-label="Delete this bill"
            data-testid="delete-bill"
            onClick={() => setConfirmDelete(true)}
            className="flex size-11 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-white/5 hover:text-destructive"
          >
            <Trash2 className="size-5" />
          </button>
        </div>

        <ClaimsBanner claims={claims} symbol={symbol} onConfirm={confirmClaim} onReject={rejectClaim} />
        <FriendNudges billId={id} symbol={symbol} />

        <ReceiptTicket title={bill.billTitle} date={bill.createdAt} symbol={symbol} rows={rows}>
          {settle && <SettleUp model={{ ...settle, payment }} />}

          <div className="space-y-3 border-t border-white/8 p-5">
            <Button type="button" variant="whatsapp" size="lg" data-testid="share-bill" onClick={share} className="w-full">
              <Share2 className="size-4" /> Share summary
            </Button>
            <button
              type="button"
              data-testid="share-image-open"
              onClick={() => setImageOpen(true)}
              className="flex min-h-12 w-full items-center justify-center gap-2 rounded-xl border border-white/10 text-base font-semibold text-foreground transition-colors hover:bg-white/5"
            >
              <ImageIcon className="size-4" /> Share as image
            </button>
            <button
              type="button"
              onClick={handleContinue}
              className="flex min-h-12 w-full items-center justify-center gap-2 rounded-xl border border-white/10 text-sm font-semibold text-foreground transition-colors hover:bg-white/5"
            >
              <Play className="size-4" fill="currentColor" /> Continue this session
            </button>
          </div>

          <div className="border-t border-white/8">
            <button
              type="button"
              aria-expanded={showReasoning}
              onClick={() => setShowReasoning(!showReasoning)}
              className="flex min-h-12 w-full items-center justify-between px-5 text-xs font-semibold text-muted-foreground transition-colors hover:bg-white/5"
            >
              <span className="flex items-center gap-2">
                <ScrollText size={14} /> How this was worked out
              </span>
              {showReasoning ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </button>
            {showReasoning && (
              <div className="px-5 pb-5">
                <div className="max-h-60 overflow-y-auto rounded-xl border border-white/8 glass-inset p-4 font-mono text-xs leading-relaxed whitespace-pre-wrap text-muted-foreground">
                  {bill.data.reasoning || "No log recorded."}
                </div>
              </div>
            )}
          </div>
        </ReceiptTicket>

        <ShareSheet
          open={imageOpen}
          onOpenChange={setImageOpen}
          data={{ title: bill.billTitle, date: bill.createdAt, symbol, shares, settlement: settle?.settlement ?? null, view }}
          payment={payment}
        />

        <ConfirmSheet
          open={confirmDelete}
          onOpenChange={setConfirmDelete}
          title="Delete this bill?"
          description="It's removed from History on all your devices. This can't be undone."
          confirmLabel="Delete"
          onConfirm={handleDelete}
        />
      </div>
    </PageTransition>
  );
}
