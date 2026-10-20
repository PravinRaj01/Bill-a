"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { BellRing, ChevronRight } from "lucide-react";

// The one Notifications row in Account > General: says whether this phone gets alerts, and opens the settings page.
export function NotificationsRow() {
  const [on, setOn] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        if (!("serviceWorker" in navigator) || !("Notification" in window) || Notification.permission !== "granted") return !cancelled && setOn(false);
        const reg = await navigator.serviceWorker.getRegistration();
        const sub = await reg?.pushManager.getSubscription();
        if (!cancelled) setOn(!!sub);
      } catch {
        if (!cancelled) setOn(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <Link
      href="/dashboard/account/notifications"
      transitionTypes={["nav-forward"]}
      data-testid="account-notifications-row"
      className="flex min-h-14 w-full items-center justify-between rounded-2xl glass px-5 text-left transition-colors hover:bg-white/10"
    >
      <span>
        <span className="block text-sm font-bold text-foreground">Notifications</span>
        <span className="block text-xs text-muted-foreground">{on === null ? " " : on ? "On for this phone" : "Off for this phone"}</span>
      </span>
      <span className="flex items-center gap-2 text-muted-foreground">
        <BellRing size={18} />
        <ChevronRight size={16} />
      </span>
    </Link>
  );
}
