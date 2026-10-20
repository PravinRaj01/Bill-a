"use client";
import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Bell } from "lucide-react";

import { getUnreadCount, markNotificationsRead } from "@/lib/actions/notifications";
import { cn } from "@/lib/utils";

// How many notifications are waiting, refreshed when you move between pages, return to the app, or the inbox
// marks something read. Guests never render this (the shell only mounts it when signed in).
export function useUnreadCount(enabled: boolean): number {
  const pathname = usePathname();
  const [count, setCount] = useState(0);
  const refresh = useCallback(() => {
    if (!enabled) return;
    getUnreadCount().then(setCount).catch(() => {
      /* offline: keep the last number */
    });
  }, [enabled]);
  useEffect(() => {
    // Opened from a phone notification: ?n=<id> says which one, so it is read now (and the bell drops with it).
    const id = new URLSearchParams(window.location.search).get("n");
    if (enabled && id && /^[0-9a-f-]{36}$/i.test(id)) {
      const url = new URL(window.location.href);
      url.searchParams.delete("n");
      window.history.replaceState(null, "", url.pathname + url.search + url.hash);
      markNotificationsRead([id]).catch(() => {}).finally(refresh);
      return;
    }
    refresh();
  }, [pathname, refresh, enabled]);
  useEffect(() => {
    const onVisible = () => document.visibilityState === "visible" && refresh();
    window.addEventListener("focus", refresh);
    window.addEventListener("notifications-changed", refresh);
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      window.removeEventListener("focus", refresh);
      window.removeEventListener("notifications-changed", refresh);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [refresh]);
  return count;
}

export const UnreadBadge = ({ count, className }: { count: number; className?: string }) =>
  count > 0 ? (
    <span data-testid="unread-badge" className={cn("flex min-w-5 items-center justify-center rounded-full bg-coral px-1.5 text-[11px] font-bold leading-5 text-coral-foreground", className)}>
      {count > 99 ? "99+" : count}
    </span>
  ) : null;

/** The header bell: a link to the inbox with the unread count on it. */
export function NotificationBell({ count }: { count: number }) {
  return (
    <Link
      href="/dashboard/notifications"
      data-testid="bell"
      aria-label={count > 0 ? `Notifications, ${count} unread` : "Notifications"}
      className="relative flex size-11 items-center justify-center rounded-full text-muted-foreground outline-none transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-white/30"
    >
      <Bell className="size-5" />
      <UnreadBadge count={count} className="absolute top-0.5 right-0 min-w-4 px-1 text-[10px] leading-4" />
    </Link>
  );
}
