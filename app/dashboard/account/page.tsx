"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { CreditCard, LogOut, Mail } from "lucide-react";

import { signOutCleanly } from "@/lib/auth/signout-client";
import { ApiKeySettings } from "@/components/ai/ApiKeySettings";
import { PageTransition } from "@/components/page-transition";
import { NotificationsRow } from "@/components/account/NotificationsRow";
import { ProfileEditor } from "@/components/account/ProfileEditor";
import { PersonAvatar } from "@/components/person-avatar";
import { useProfile } from "@/components/profile-provider";
import { PaymentDetailsForm } from "@/components/payment/PaymentDetailsForm";
import { Segmented } from "@/components/ui/segmented";
import { usePaymentDetails } from "@/hooks/use-payment-details";
import { getCurrentUser } from "@/lib/actions/user";

type Tab = "overview" | "profile" | "pay" | "ai";

const TABS: Tab[] = ["overview", "profile", "pay", "ai"];
const asTab = (v: string | null): Tab | null => (TABS as string[]).includes(v ?? "") ? (v as Tab) : null;

// useSearchParams needs a Suspense boundary (the page is prerendered), hence the thin wrapper.
export default function AccountPage() {
  return (
    <Suspense fallback={null}>
      <AccountContent />
    </Suspense>
  );
}

function AccountContent() {
  const [user, setUser] = useState<Awaited<ReturnType<typeof getCurrentUser>>>(null);
  // The Account page is sign-in only, so payment details come from the account.
  const payment = usePaymentDetails(true);
  const profile = useProfile();
  // The page is split into tabs so a phone never has to scroll through everything at once.
  // Links open straight on a tab (?tab=pay from the Home checklist), and
  // following another such link while already on this page switches tab too, so the URL is followed, not just read once.
  const search = useSearchParams();
  const urlTab = search.get("tab");
  const guide = search.get("guide") === "1"; // the welcome notice opens the AI key steps
  const [tab, setTab] = useState<Tab>(asTab(urlTab) ?? "overview");
  const [seenUrlTab, setSeenUrlTab] = useState(urlTab);
  if (urlTab !== seenUrlTab) {
    setSeenUrlTab(urlTab);
    const next = asTab(urlTab);
    if (next) setTab(next);
  }

  useEffect(() => {
    getCurrentUser().then(setUser);
  }, []);

  return (
    <PageTransition>
      <div className="mx-auto mb-20 max-w-xl space-y-5 p-6 text-foreground md:p-10">
        <header className="space-y-1">
          <h1 className="type-title">My Account</h1>
        </header>

        <Segmented
          label="Account sections"
          value={tab}
          onValueChange={setTab}
          options={[
            { value: "overview", label: "General" },
            { value: "profile", label: "Profile" },
            { value: "pay", label: "Get paid" },
            { value: "ai", label: "AI" },
          ]}
        />

        {tab === "overview" && (
          <div data-testid="account-overview" className="space-y-4">
            <div className="flex items-center gap-4 rounded-3xl glass p-5">
              <PersonAvatar name={profile.meName} className="size-16" />
              <div className="min-w-0 space-y-1">
                <h2 className="truncate text-lg font-bold tracking-tight text-foreground">{profile.name || user?.name || "Bill.a User"}</h2>
                {profile.username ? (
                  <p data-testid="account-username" className="truncate text-sm text-primary">@{profile.username}</p>
                ) : (
                  <button type="button" onClick={() => setTab("profile")} className="text-sm text-muted-foreground underline underline-offset-4 hover:text-foreground">
                    Choose a username
                  </button>
                )}
                <p className="flex items-center gap-1.5 truncate font-mono text-xs text-muted-foreground">
                  <Mail size={12} className="shrink-0" />
                  <span className="truncate">{user?.email}</span>
                </p>
              </div>
            </div>

            <button
              type="button"
              onClick={() => setTab("pay")}
              className="flex min-h-14 w-full items-center justify-between rounded-2xl glass px-5 text-left transition-colors hover:bg-white/10"
            >
              <span>
                <span className="block text-sm font-bold text-foreground">Get paid</span>
                <span className="block text-xs text-muted-foreground">{payment.details ? "Payment details saved" : "Add how friends can pay you back"}</span>
              </span>
              <CreditCard size={18} className="text-muted-foreground" />
            </button>

            <NotificationsRow />

            <button
              type="button"
              onClick={() => void signOutCleanly()}
              className="group flex min-h-14 w-full items-center justify-between rounded-2xl glass px-5 text-xs font-bold tracking-widest uppercase transition-colors hover:bg-white/10"
            >
              <span>Sign Out</span>
              <LogOut size={16} className="text-muted-foreground transition-colors group-hover:text-foreground" />
            </button>

            <p className="text-center font-mono text-xs text-muted-foreground/60 uppercase">User ID: {user?.id?.slice(0, 8)}...</p>
          </div>
        )}

        {tab === "profile" && <ProfileEditor />}

        {tab === "pay" && (
          <section className="space-y-3" data-testid="account-payment">
            <p className="px-1 text-sm text-muted-foreground">Add how friends can pay you back. It&apos;s added to the messages and receipts you share, only when you choose to.</p>
            {payment.details === undefined ? (
              <div className="h-48 animate-pulse rounded-3xl glass" aria-label="Loading payment details" />
            ) : (
              <PaymentDetailsForm initial={payment.details} onSave={payment.save} />
            )}
          </section>
        )}

        {tab === "ai" && (
          <section className="space-y-3" data-testid="account-ai-settings">
            <ApiKeySettings guideOpen={guide} />
          </section>
        )}
      </div>
    </PageTransition>
  );
}
