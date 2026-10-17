"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { toast } from "sonner";

import { BalanceCard } from "@/components/balances/BalanceCard";
import { PageTransition } from "@/components/page-transition";
import { PaymentToggle } from "@/components/payment/PaymentToggle";
import { usePaymentShare } from "@/hooks/use-payment-share";
import { getBalances, getBill, type BalancesResult } from "@/lib/actions/history";
import { pushOutbox } from "@/lib/actions/sync";
import { getCurrentUser } from "@/lib/actions/user";
import { formatMoney } from "@/lib/money";
import { balanceNudgeText, localDate, markPersonPaid, type PersonBalance } from "@/lib/split/settle";
import { payloadWithSettlement } from "@/lib/sync/billPayload";
import { enqueueBill, flushOutbox } from "@/lib/sync/outbox";
import { shareMessage, shareToPerson } from "@/lib/ui/share";
import { usePhoneBook } from "@/hooks/use-phone-book";
import { Button } from "@/components/ui/button";

// What each person owes you (or you them) across every bill you track, so "how much does Riley owe me?"
// has one answer. Nudging sends one message listing what they owe per bill; Mark all paid updates every
// bill's settlement through the outbox (the same rows, never duplicates) and can be undone.
export default function BalancesPage() {
  const [result, setResult] = useState<BalancesResult | null>(null);
  const [user, setUser] = useState<{ id: string } | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const payment = usePaymentShare();
  const { phoneFor } = usePhoneBook();

  const reload = useCallback(async () => {
    setResult(await getBalances());
  }, []);

  useEffect(() => {
    getCurrentUser().then((u) => setUser(u ? { id: u.id } : null)).catch(() => setUser(null));
    getBalances().then(setResult).catch(() => setResult({ people: [], tracked: 0, withoutMe: 0 }));
  }, []);

  const totals = useMemo(() => {
    const byCurrency = new Map<string, { owedToYou: number; youOwe: number }>();
    for (const p of result?.people ?? []) {
      const t = byCurrency.get(p.currency) ?? { owedToYou: 0, youOwe: 0 };
      if (p.net > 0) t.owedToYou += p.net;
      else t.youOwe += -p.net;
      byCurrency.set(p.currency, t);
    }
    return [...byCurrency];
  }, [result]);

  /** Push every pending save now and wait, so the next read (and the next page) sees the result. */
  const syncNow = async (userId: string) => {
    try {
      await flushOutbox(userId, pushOutbox);
    } catch {
      /* offline: it stays in the outbox and syncs later */
    }
  };

  const nudge = async (p: PersonBalance) => {
    const text = balanceNudgeText({
      name: p.name,
      payTo: "me",
      currency: p.currency,
      lines: p.bills.filter((b) => b.amount > 0).map((b) => ({ title: b.title, date: b.createdAt, amount: b.amount })),
      payment: payment.block || undefined, // these balances are owed to the user
    });
    const msg = shareMessage(await shareToPerson(text, phoneFor(p.name)));
    if (msg) toast(msg);
  };

  const settle = async (p: PersonBalance) => {
    if (!user) return;
    setBusy(p.key);
    try {
      const today = localDate();
      const before: Awaited<ReturnType<typeof getBill>>[] = [];
      for (const b of p.bills) {
        const row = await getBill(b.id);
        const current = row?.data.settlement;
        if (!row || !current) continue;
        const next = markPersonPaid(current, row.data.split, p.name, today);
        if (Object.keys(next.paid).length <= Object.keys(current.paid).length) continue; // nothing new to mark
        await enqueueBill(user.id, payloadWithSettlement(row, next));
        before.push(row);
      }
      await syncNow(user.id);
      await reload();
      if (before.length === 0) return;
      toast(`Marked ${before.length} ${before.length === 1 ? "bill" : "bills"} paid with ${p.name}`, {
        action: {
          label: "Undo",
          onClick: async () => {
            for (const row of before) if (row) await enqueueBill(user.id, payloadWithSettlement(row, row.data.settlement ?? null));
            await syncNow(user.id);
            await reload();
          },
        },
      });
    } catch (e) {
      console.error("Mark all paid failed", e);
      toast.error("Couldn't update those bills. Please try again.");
    } finally {
      setBusy(null);
    }
  };

  const people = result?.people ?? [];

  return (
    <PageTransition>
      <div className="mx-auto mb-20 max-w-4xl space-y-6 p-6 text-foreground md:p-10">
        <header className="space-y-1">
          <h1 className="type-title">Balances</h1>
          <p className="type-eyebrow">Who owes what, across bills</p>
        </header>

        {result === null ? (
          <div className="h-40 animate-pulse rounded-3xl glass" aria-label="Loading balances" />
        ) : (
          <>
            {totals.length > 0 && (
              <div data-testid="balances-total" className="space-y-1 rounded-2xl glass p-4">
                {totals.map(([cur, t]) => (
                  <div key={cur} className="space-y-0.5 text-sm text-muted-foreground">
                    {t.owedToYou > 0 && (
                      <p>
                        You&apos;re owed <strong className="text-lg font-bold text-success tabular-nums">{formatMoney(t.owedToYou, cur)}</strong>
                      </p>
                    )}
                    {t.youOwe > 0 && (
                      <p>
                        You owe <strong className="text-lg font-bold text-coral tabular-nums">{formatMoney(t.youOwe, cur)}</strong>
                      </p>
                    )}
                  </div>
                ))}
              </div>
            )}

            {people.length > 0 && <PaymentToggle payment={payment} />}

            {people.length > 0 ? (
              <ul className="grid gap-4 lg:grid-cols-2 lg:items-start">
                {people.map((p) => (
                  <BalanceCard key={p.key} person={p} busy={busy === p.key} onNudge={nudge} onSettle={settle} />
                ))}
              </ul>
            ) : (
              <div data-testid="balances-empty" className="space-y-3 rounded-3xl border-2 border-dashed border-white/8 px-6 py-16 text-center">
                {result.tracked === 0 ? (
                  <>
                    <p className="text-base font-semibold text-foreground">Nothing tracked yet</p>
                    <p className="mx-auto max-w-sm text-sm text-muted-foreground">
                      On a bill&apos;s Settle up card, say who paid or mark someone as paid, and what people owe you shows up here.
                    </p>
                    <Button asChild>
                      <Link href="/dashboard/new">Start a split</Link>
                    </Button>
                  </>
                ) : result.withoutMe === result.tracked ? (
                  <>
                    <p className="text-base font-semibold text-foreground">Which one are you?</p>
                    <p className="mx-auto max-w-sm text-sm text-muted-foreground">
                      Your tracked bills don&apos;t say which person is you, so there&apos;s nothing to add up yet. Open one in History and choose &ldquo;Which one is you?&rdquo;.
                    </p>
                    <Link href="/dashboard/history" className="inline-flex min-h-11 items-center rounded-full border border-white/15 px-5 text-sm font-semibold text-foreground">
                      Open History
                    </Link>
                  </>
                ) : (
                  <>
                    <p className="text-base font-semibold text-success">You&apos;re all square</p>
                    <p className="text-sm text-muted-foreground">Nobody owes you anything, and you owe nobody.</p>
                  </>
                )}
              </div>
            )}

            {people.length > 0 && result.withoutMe > 0 && (
              <p data-testid="without-me" className="px-1 text-sm text-muted-foreground">
                {result.withoutMe} tracked {result.withoutMe === 1 ? "bill doesn't" : "bills don't"} say which person is you, so {result.withoutMe === 1 ? "it isn't" : "they aren't"} counted.{" "}
                <Link href="/dashboard/history" className="font-semibold text-foreground underline underline-offset-4">
                  Open History
                </Link>{" "}
                and choose &ldquo;Which one is you?&rdquo;.
              </p>
            )}
          </>
        )}
      </div>
    </PageTransition>
  );
}
