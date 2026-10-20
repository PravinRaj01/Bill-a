"use client";
import Link from "next/link";
import { ChevronLeft } from "lucide-react";

import { NotificationPrefs } from "@/components/account/NotificationPrefs";
import { NotificationSettings } from "@/components/account/NotificationSettings";
import { PageTransition } from "@/components/page-transition";

// Notification settings: whether this phone gets alerts at all (and a test), then which kinds reach it.
export default function NotificationSettingsPage() {
  return (
    <PageTransition>
      <div className="mx-auto mb-20 min-h-screen max-w-xl space-y-6 p-6 text-foreground md:p-10">
        <Link href="/dashboard/account" transitionTypes={["nav-back"]} className="flex min-h-11 items-center text-sm font-semibold text-muted-foreground transition-colors hover:text-foreground">
          <ChevronLeft size={16} className="mr-1" /> Account
        </Link>

        <header className="space-y-1">
          <h1 className="type-title">Notifications</h1>
          <p className="type-eyebrow">What you hear about, and where</p>
        </header>

        <NotificationSettings />
        <NotificationPrefs />
      </div>
    </PageTransition>
  );
}
