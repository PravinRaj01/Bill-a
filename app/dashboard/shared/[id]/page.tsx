"use client";
import { useCallback, useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { Check, ChevronLeft, EyeOff, Loader2 } from "lucide-react";
import { QRCodeSVG } from "qrcode.react";
import { toast } from "sonner";

import { PageTransition } from "@/components/page-transition";
import { PersonAvatar } from "@/components/person-avatar";
import { ReceiptTicket, type TicketRow } from "@/components/receipt/ReceiptTicket";
import { Button } from "@/components/ui/button";
import { ConfirmSheet } from "@/components/ui/confirm-sheet";
import { DuplicateNotice } from "@/components/history/DuplicateNotice";
import { claimPaid, getSharedDetail, hideShared, listDuplicates, type DuplicateView, type SharedDetail } from "@/lib/actions/shared";
import { formatMoney } from "@/lib/money";

// A bill a friend shared with you: read-only. You can see the split, say "I've paid" for what you owe the
// friend (they confirm or reject it), or hide the bill from your History.
export default function SharedBillPage() {
  const params = useParams();
  const id = String(params?.id ?? "");
  const router = useRouter();
  const [bill, setBill] = useState<SharedDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [confirmHide, setConfirmHide] = useState(false);
  const [dup, setDup] = useState<DuplicateView | null>(null);

  const load = useCallback(async () => {
    const d = await getSharedDetail(id).catch(() => null);
    if (!d) {
      toast("This bill isn't shared with you any more");
      router.push("/dashboard/history");
    }
    else setBill(d);
    setLoading(false);
  }, [id, router]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- load from the server on mount
    void load();
    listDuplicates().then((all) => setDup(all.find((d) => d.sharedId === id) ?? null)).catch(() => setDup(null));
  }, [load, id]);

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-background text-foreground">
        <Loader2 className="animate-spin" />
      </div>
    );
  }
  if (!bill) return null;

  const symbol = bill.currency || "RM";
  const rows: TicketRow[] = bill.split.map((r) => ({ name: r.name, amount: r.amount, items: r.items }));

  const say = async (key: string) => {
    setBusy(key);
    try {
      const r = await claimPaid(bill.id, key);
      if (r.ok) toast.success(`Told ${bill.owner.name}. They'll confirm it.`);
      else toast.error(r.error);
      await load();
    } catch {
      toast.error("Couldn't send that. Please try again.");
    } finally {
      setBusy(null);
    }
  };

  const hide = async () => {
    await hideShared(bill.id, true);
    toast("Hidden from your History");
    router.push("/dashboard/history", { transitionTypes: ["nav-back"] });
    router.refresh();
  };

  const mine = bill.open;
  return (
    <PageTransition>
      <div className="mx-auto mb-20 max-w-2xl space-y-6 p-6 text-foreground md:p-10">
        <div className="flex items-center justify-between">
          <button type="button" onClick={() => router.push("/dashboard/history", { transitionTypes: ["nav-back"] })} className="flex min-h-11 items-center pr-3 text-sm font-semibold text-muted-foreground transition-colors hover:text-foreground">
            <ChevronLeft className="mr-1 size-4" /> Back
          </button>
          <button type="button" aria-label="Hide this bill" data-testid="hide-shared" onClick={() => setConfirmHide(true)} className="flex size-11 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-white/5 hover:text-foreground">
            <EyeOff className="size-5" />
          </button>
        </div>

        <div data-testid="shared-by" className="flex items-center gap-3 rounded-3xl glass p-4">
          <PersonAvatar name={bill.owner.name} seed={bill.owner.avatarSeed} className="size-11" />
          <div className="min-w-0">
            <p className="text-sm font-bold">Shared by {bill.owner.name}</p>
            <p className="text-xs text-muted-foreground">You&apos;re {bill.asName} on this bill. Only {bill.owner.name} can change it.</p>
          </div>
        </div>

        {dup && <DuplicateNotice dup={dup} viewing="theirs" />}

        <ReceiptTicket title={bill.billTitle} date={bill.createdAt} symbol={symbol} rows={rows}>
          <div className="space-y-3 border-t border-white/8 p-5" data-testid="shared-settle">
            {mine.length === 0 ? (
              <p data-testid="shared-settled" className="flex items-center gap-2 text-sm font-semibold text-success">
                <Check size={16} strokeWidth={3} /> Nothing owed between you and {bill.owner.name} on this bill.
              </p>
            ) : (
              mine.map((o) =>
                o.direction === "iOwe" ? (
                  <div key={o.key} data-testid="shared-owe" className="space-y-3 rounded-2xl glass-inset p-4">
                    <p className="text-sm">
                      You owe <strong className="text-coral tabular-nums">{formatMoney(o.amount, symbol)}</strong> to {bill.owner.name}
                    </p>
                    {o.claim === "pending" ? (
                      <p data-testid="claim-pending" className="text-sm font-semibold text-muted-foreground">Waiting for {bill.owner.name} to confirm</p>
                    ) : (
                      <>
                        {o.claim === "rejected" && (
                          <p data-testid="claim-rejected" className="text-sm text-warning">
                            {bill.owner.name} says they haven&apos;t received it. Send it again once they have.
                          </p>
                        )}
                        <Button type="button" data-testid="claim-paid" variant="whatsapp" size="lg" className="w-full" disabled={busy === o.key} onClick={() => say(o.key)}>
                          {busy === o.key ? <Loader2 className="size-4 animate-spin" /> : <Check className="mr-2 size-4" strokeWidth={3} />} I&apos;ve paid
                        </Button>
                      </>
                    )}
                  </div>
                ) : (
                  <p key={o.key} className="text-sm">
                    {bill.owner.name} owes you <strong className="text-success tabular-nums">{formatMoney(o.amount, symbol)}</strong> on this bill.
                  </p>
                ),
              )
            )}

            {bill.owesOwner && (bill.payLines.length > 0 || bill.payQr) && (
              <div data-testid="shared-pay" className="space-y-2 rounded-2xl glass-inset p-4">
                <p className="type-eyebrow">How to pay {bill.owner.name}</p>
                {bill.payLines.map((l) => (
                  <p key={l} className="text-sm">{l}</p>
                ))}
                {bill.payQr && (
                  <div className="inline-block rounded-2xl bg-white p-2">
                    <QRCodeSVG value={bill.payQr.payload} size={140} level="M" marginSize={1} title={`${bill.owner.name}'s DuitNow QR`} />
                  </div>
                )}
              </div>
            )}
          </div>
        </ReceiptTicket>

        <ConfirmSheet
          open={confirmHide}
          onOpenChange={setConfirmHide}
          title="Hide this bill?"
          description={`It's removed from your History and balances. You can bring it back from the shared list in History.`}
          confirmLabel="Hide"
          onConfirm={hide}
        />
      </div>
    </PageTransition>
  );
}
