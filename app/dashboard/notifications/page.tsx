"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Bell, CheckCheck, ChevronLeft } from "lucide-react";

import { PageTransition } from "@/components/page-transition";
import { PersonAvatar } from "@/components/person-avatar";
import { Button } from "@/components/ui/button";
import { getCurrentUser } from "@/lib/actions/user";
import { getInbox, markAllNotificationsRead, markNotificationsRead, type InboxEntry } from "@/lib/actions/notifications";
import { cn } from "@/lib/utils";

const ago = (iso: string): string => {
  const mins = Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 60000));
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  return days < 7 ? `${days}d ago` : new Date(iso).toLocaleDateString(undefined, { day: "numeric", month: "short" });
};

const changed = () => window.dispatchEvent(new Event("notifications-changed"));

// What has happened to you: friend requests, bills friends added you to, payments, reminders. Tapping one marks
// it read and opens what it is about.
export default function NotificationsPage() {
  const router = useRouter();
  const [items, setItems] = useState<InboxEntry[] | null>(null);
  const [signedIn, setSignedIn] = useState<boolean | null>(null);

  useEffect(() => {
    getCurrentUser().then((u) => setSignedIn(!!u)).catch(() => setSignedIn(false));
    getInbox().then(setItems).catch(() => setItems([]));
  }, []);

  const open = async (n: InboxEntry) => {
    if (!n.read) {
      setItems((all) => (all ?? []).map((x) => (x.id === n.id ? { ...x, read: true } : x)));
      markNotificationsRead([n.id]).then(changed).catch(() => {});
    }
    if (n.href) router.push(n.href, { transitionTypes: ["nav-forward"] });
  };

  const readAll = async () => {
    setItems((all) => (all ?? []).map((x) => ({ ...x, read: true })));
    await markAllNotificationsRead().catch(() => {});
    changed();
  };

  const unread = (items ?? []).filter((n) => !n.read).length;

  return (
    <PageTransition>
      <div className="mx-auto mb-20 min-h-screen max-w-xl space-y-6 p-6 text-foreground md:p-10">
        <Link href="/dashboard" transitionTypes={["nav-back"]} className="flex min-h-11 items-center text-sm font-semibold text-muted-foreground transition-colors hover:text-foreground">
          <ChevronLeft size={16} className="mr-1" /> Home
        </Link>

        <header className="flex items-end justify-between gap-3">
          <div className="space-y-1">
            <h1 className="type-title">Notifications</h1>
            <p className="type-eyebrow">{unread > 0 ? `${unread} new` : "All caught up"}</p>
          </div>
          {unread > 0 && (
            <Button type="button" variant="glass" size="sm" data-testid="mark-all-read" onClick={() => void readAll()}>
              <CheckCheck /> Mark all read
            </Button>
          )}
        </header>

        {items && items.length > 0 && (
          <ul className="space-y-2">
            {items.map((n) => (
              <li key={n.id}>
                <button
                  type="button"
                  data-testid="notification"
                  data-read={n.read}
                  onClick={() => void open(n)}
                  className={cn("flex min-h-16 w-full items-center gap-3 rounded-3xl p-4 text-left transition-colors hover:bg-white/10", n.read ? "glass opacity-70" : "glass border border-primary/30")}
                >
                  <PersonAvatar name={n.actor?.name ?? "Bill.a"} seed={n.actor?.avatarSeed} className="size-11" />
                  <span className="min-w-0 flex-1">
                    <span className={cn("block text-sm", n.read ? "font-medium" : "font-bold")}>{n.text}</span>
                    <span className="block text-xs text-muted-foreground">{ago(n.createdAt)}</span>
                  </span>
                  {!n.read && <span aria-label="Unread" className="size-2.5 shrink-0 rounded-full bg-coral" />}
                </button>
              </li>
            ))}
          </ul>
        )}

        {items && items.length === 0 && (
          <div className="space-y-3 rounded-3xl border-2 border-dashed border-white/8 px-6 py-14 text-center">
            <Bell className="mx-auto size-9 text-muted-foreground/50" />
            {signedIn === false ? (
              <p className="text-sm text-muted-foreground">
                <Link href="/" className="font-semibold text-foreground underline underline-offset-4">Log in</Link> to get friend requests, shared bills and payment updates here.
              </p>
            ) : (
              <>
                <p className="text-base font-semibold text-foreground">Nothing yet</p>
                <p className="mx-auto max-w-xs text-sm text-muted-foreground">Friend requests, bills friends add you to and payment updates show up here.</p>
              </>
            )}
          </div>
        )}
      </div>
    </PageTransition>
  );
}
