"use client";

import { useEffect, useState } from "react";
import { Bot, CreditCard, History, LogOut, Mail, User, Wallet } from "lucide-react";

import { signOutAction } from "@/app/auth/actions";
import { ApiKeySettings } from "@/components/ai/ApiKeySettings";
import { PageTransition } from "@/components/page-transition";
import { PaymentDetailsForm } from "@/components/payment/PaymentDetailsForm";
import { Segmented } from "@/components/ui/segmented";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { usePaymentDetails } from "@/hooks/use-payment-details";
import { getAccountStats } from "@/lib/actions/history";
import { getCurrentUser } from "@/lib/actions/user";
import { fromCents } from "@/lib/money";

type Tab = "overview" | "pay" | "ai";

/** Whole units with the bill's own currency, e.g. "RM1,234". */
const whole = (currency: string, cents: number) => `${currency}${Math.round(fromCents(cents)).toLocaleString()}`;

export default function AccountPage() {
  const [user, setUser] = useState<Awaited<ReturnType<typeof getCurrentUser>>>(null);
  const [stats, setStats] = useState<Awaited<ReturnType<typeof getAccountStats>>>({ count: 0, totals: [] });
  // The Account page is sign-in only, so payment details come from the account.
  const payment = usePaymentDetails(true);
  // The page is split into tabs so a phone never has to scroll through everything at once.
  const [tab, setTab] = useState<Tab>("overview");

  useEffect(() => {
    Promise.all([getCurrentUser(), getAccountStats()]).then(([u, s]) => {
      setUser(u);
      setStats(s);
    });
  }, []);

  const [biggest, ...others] = stats.totals;

  return (
    <PageTransition>
      <div className="mx-auto mb-20 max-w-xl space-y-5 bg-background p-6 text-white md:p-10">
        <header className="space-y-1">
          <h1 className="text-3xl font-black tracking-tighter text-white uppercase italic">My Account</h1>
        </header>

        <Segmented
          label="Account sections"
          value={tab}
          onValueChange={setTab}
          options={[
            { value: "overview", label: "Overview", icon: <User className="size-4" /> },
            { value: "pay", label: "Get paid", icon: <CreditCard className="size-4" /> },
            { value: "ai", label: "AI", icon: <Bot className="size-4" /> },
          ]}
        />

        {tab === "overview" && (
          <div data-testid="account-overview" className="space-y-4">
            <div className="flex items-center gap-4 rounded-3xl border border-white/5 bg-surface p-5">
              <div className="size-16 shrink-0 rounded-full bg-surface-2 p-1 ring-1 ring-white/10">
                <Avatar className="size-full rounded-full">
                  <AvatarImage src={user?.image ?? undefined} />
                  <AvatarFallback className="bg-background text-xl font-bold text-muted-foreground">{user?.email?.charAt(0).toUpperCase()}</AvatarFallback>
                </Avatar>
              </div>
              <div className="min-w-0 space-y-1">
                <h2 className="truncate text-lg font-bold tracking-tight text-white">{user?.name || "Bill.a User"}</h2>
                <p className="flex items-center gap-1.5 truncate font-mono text-xs text-muted-foreground">
                  <Mail size={12} className="shrink-0" />
                  <span className="truncate">{user?.email}</span>
                </p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="relative overflow-hidden rounded-3xl border border-white/5 bg-surface p-5">
                <History size={44} className="absolute top-3 right-3 opacity-10" />
                <div className="relative space-y-1">
                  <span className="text-xs font-bold tracking-widest text-muted-foreground uppercase">Sessions</span>
                  <div className="text-3xl font-black tracking-tighter text-white tabular-nums">{stats.count}</div>
                </div>
              </div>
              <div data-testid="account-total" className="relative overflow-hidden rounded-3xl border border-white/5 bg-surface p-5">
                <Wallet size={44} className="absolute top-3 right-3 opacity-10" />
                <div className="relative space-y-1">
                  <span className="text-xs font-bold tracking-widest text-muted-foreground uppercase">Total value</span>
                  <div className="text-3xl font-black tracking-tighter text-white tabular-nums">{biggest ? whole(biggest.currency, biggest.cents) : "0"}</div>
                  {/* Bills in other currencies are listed, never added into the big number. */}
                  {others.length > 0 && <p className="text-xs font-semibold text-muted-foreground tabular-nums">{others.map((t) => whole(t.currency, t.cents)).join(" · ")}</p>}
                </div>
              </div>
            </div>

            <button
              type="button"
              onClick={() => setTab("pay")}
              className="flex min-h-14 w-full items-center justify-between rounded-2xl border border-white/5 bg-surface px-5 text-left transition-colors hover:bg-surface-2"
            >
              <span>
                <span className="block text-sm font-bold text-white">Get paid</span>
                <span className="block text-xs text-muted-foreground">{payment.details ? "Payment details saved" : "Add how friends can pay you back"}</span>
              </span>
              <CreditCard size={18} className="text-muted-foreground" />
            </button>

            <button
              type="button"
              onClick={() => signOutAction()}
              className="group flex min-h-14 w-full items-center justify-between rounded-2xl border border-white/5 bg-surface px-5 text-xs font-bold tracking-widest uppercase transition-colors hover:bg-surface-2"
            >
              <span>Sign Out</span>
              <LogOut size={16} className="text-muted-foreground transition-colors group-hover:text-white" />
            </button>

            <p className="text-center font-mono text-xs text-muted-foreground/60 uppercase">User ID: {user?.id?.slice(0, 8)}...</p>
          </div>
        )}

        {tab === "pay" && (
          <section className="space-y-3" data-testid="account-payment">
            <p className="px-1 text-sm text-muted-foreground">Add how friends can pay you back. It&apos;s added to the messages and receipts you share, only when you choose to.</p>
            {payment.details === undefined ? (
              <div className="h-48 animate-pulse rounded-3xl bg-surface" aria-label="Loading payment details" />
            ) : (
              <PaymentDetailsForm initial={payment.details} onSave={payment.save} />
            )}
          </section>
        )}

        {tab === "ai" && (
          <section className="space-y-3" data-testid="account-ai-settings">
            <ApiKeySettings />
          </section>
        )}
      </div>
    </PageTransition>
  );
}
