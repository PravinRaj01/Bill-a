"use client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { AnimatePresence, m } from "motion/react";
import { Bell, CheckCheck, ChevronLeft, Trash2, X } from "lucide-react";
import { toast } from "sonner";

import { PageTransition } from "@/components/page-transition";
import { PersonAvatar } from "@/components/person-avatar";
import { Button } from "@/components/ui/button";
import { getCurrentUser } from "@/lib/actions/user";
import {
  clearAllNotifications,
  dismissNotifications,
  getInbox,
  markAllNotificationsRead,
  markNotificationsRead,
  restoreNotifications,
  type InboxEntry,
} from "@/lib/actions/notifications";
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
const SWIPE = 90; // px to the left that clears a notification

// What has happened to you: friend requests, bills friends added you to, payments, reminders. Tapping one marks
// it read and opens what it is about; swiping it left (or its X) clears it, and Clear all empties the inbox. Every
// clear can be undone for a few seconds.
export default function NotificationsPage() {
  const router = useRouter();
  const [items, setItems] = useState<InboxEntry[] | null>(null);
  const [signedIn, setSignedIn] = useState<boolean | null>(null);
  // Letting go of a swipe also lands on the row as a click: that must not open it.
  const dragging = useRef(false);
  const [dragId, setDragId] = useState<string | null>(null); // the row being swiped: only then is the clear hint shown

  const load = useCallback(() => getInbox().then(setItems).catch(() => setItems((s) => s ?? [])), []);

  useEffect(() => {
    getCurrentUser().then((u) => setSignedIn(!!u)).catch(() => setSignedIn(false));
    // eslint-disable-next-line react-hooks/set-state-in-effect -- load from the server on mount
    void load();
    // A push that arrives while the inbox is open shows up without a reload.
    window.addEventListener("notifications-changed", load);
    return () => window.removeEventListener("notifications-changed", load);
  }, [load]);

  const open = async (n: InboxEntry) => {
    if (dragging.current) return;
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

  const undo = async (ids: string[]) => {
    try {
      await restoreNotifications(ids);
      await load();
      changed();
    } catch {
      toast.error("Couldn't bring those back.");
    }
  };

  const clearOne = async (n: InboxEntry) => {
    setItems((all) => (all ?? []).filter((x) => x.id !== n.id));
    try {
      const done = await dismissNotifications([n.id]);
      changed();
      toast("Cleared", { action: { label: "Undo", onClick: () => void undo(done) } });
    } catch {
      toast.error("Couldn't clear that. Please try again.");
      void load();
    }
  };

  const clearAll = async () => {
    const before = items;
    setItems([]);
    try {
      const done = await clearAllNotifications();
      changed();
      toast(`Cleared ${done.length}`, { action: { label: "Undo", onClick: () => void undo(done) } });
    } catch {
      toast.error("Couldn't clear those. Please try again.");
      setItems(before);
    }
  };

  const unread = (items ?? []).filter((n) => !n.read).length;

  // Today first, then everything earlier (by the reader's own calendar day).
  const groups = useMemo(() => {
    const today = new Date().toDateString();
    const list = items ?? [];
    return [
      { label: "Today", rows: list.filter((n) => new Date(n.createdAt).toDateString() === today) },
      { label: "Earlier", rows: list.filter((n) => new Date(n.createdAt).toDateString() !== today) },
    ].filter((g) => g.rows.length > 0);
  }, [items]);

  return (
    <PageTransition>
      <div className="mx-auto mb-20 min-h-screen max-w-xl space-y-6 p-6 text-foreground md:p-10">
        <Link href="/dashboard" transitionTypes={["nav-back"]} className="flex min-h-11 items-center text-sm font-semibold text-muted-foreground transition-colors hover:text-foreground">
          <ChevronLeft size={16} className="mr-1" /> Home
        </Link>

        <header className="space-y-3">
          <div className="space-y-1">
            <h1 className="type-title">Notifications</h1>
            <p className="type-eyebrow">{items === null ? " " : unread > 0 ? `${unread} new` : "All caught up"}</p>
          </div>
          <div className="flex gap-2">
            {unread > 0 && (
              <Button type="button" variant="glass" size="sm" data-testid="mark-all-read" onClick={() => void readAll()}>
                <CheckCheck /> Read all
              </Button>
            )}
            {items && items.length > 0 && (
              <Button type="button" variant="glass" size="sm" data-testid="clear-all" onClick={() => void clearAll()}>
                <Trash2 /> Clear all
              </Button>
            )}
          </div>
        </header>

        {items === null && (
          <ul aria-label="Loading notifications" className="space-y-2">
            {[0, 1, 2].map((i) => (
              <li key={i} className="h-16 animate-pulse rounded-3xl glass" />
            ))}
          </ul>
        )}

        {groups.map((g) => (
          <section key={g.label} aria-label={g.label} className="space-y-2">
            <h2 className="type-eyebrow px-1">{g.label}</h2>
            <ul className="space-y-2">
              <AnimatePresence initial={false}>
                {g.rows.map((n) => (
                  <m.li key={n.id} layout="position" exit={{ opacity: 0, height: 0, marginTop: 0 }} transition={{ duration: 0.2 }} className="relative overflow-hidden rounded-3xl">
                    {/* Revealed as the row is dragged left. */}
                    <div aria-hidden style={{ opacity: dragId === n.id ? 1 : 0 }} className="absolute inset-0 flex items-center justify-end rounded-3xl bg-destructive/25 pr-6 text-sm font-semibold text-destructive transition-opacity">
                      <Trash2 className="size-5" />
                    </div>
                    <m.div
                      drag="x"
                      dragDirectionLock
                      dragConstraints={{ left: 0, right: 0 }}
                      dragElastic={{ left: 0.6, right: 0 }}
                      onDragStart={() => {
                        dragging.current = true;
                        setDragId(n.id);
                      }}
                      onDragEnd={(_, info) => {
                        setDragId(null);
                        setTimeout(() => {
                          dragging.current = false;
                        }, 80);
                        if (info.offset.x < -SWIPE || info.velocity.x < -600) void clearOne(n);
                      }}
                      className={cn("relative flex touch-pan-y items-center gap-1 rounded-3xl bg-background/60", n.read ? "glass opacity-70" : "glass border border-primary/30")}
                    >
                      <button
                        type="button"
                        data-testid="notification"
                        data-read={n.read}
                        onClick={() => void open(n)}
                        className="flex min-h-16 min-w-0 flex-1 items-center gap-3 rounded-3xl p-4 text-left transition-colors hover:bg-white/10"
                      >
                        <PersonAvatar name={n.actor?.name ?? "Bill.a"} seed={n.actor?.avatarSeed} className="size-11" />
                        <span className="min-w-0 flex-1">
                          <span className={cn("block text-sm", n.read ? "font-medium" : "font-bold")}>{n.text}</span>
                          <span className="block text-xs text-muted-foreground">{ago(n.createdAt)}</span>
                        </span>
                        {!n.read && <span aria-label="Unread" className="size-2.5 shrink-0 rounded-full bg-coral" />}
                      </button>
                      <button
                        type="button"
                        data-testid="clear-one"
                        aria-label="Clear this notification"
                        onClick={() => void clearOne(n)}
                        className="mr-2 flex size-10 shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-white/10 hover:text-foreground"
                      >
                        <X size={16} />
                      </button>
                    </m.div>
                  </m.li>
                ))}
              </AnimatePresence>
            </ul>
          </section>
        ))}

        {items && items.length === 0 && (
          <div className="space-y-3 rounded-3xl border-2 border-dashed border-white/8 px-6 py-14 text-center">
            <Bell className="mx-auto size-9 text-muted-foreground/50" />
            {signedIn === false ? (
              <p className="text-sm text-muted-foreground">
                <Link href="/" className="font-semibold text-foreground underline underline-offset-4">Log in</Link> to get friend requests, shared bills and payment updates here.
              </p>
            ) : (
              <>
                <p className="text-base font-semibold text-foreground">All clear</p>
                <p className="mx-auto max-w-xs text-sm text-muted-foreground">Friend requests, bills friends add you to and payment updates show up here.</p>
              </>
            )}
          </div>
        )}
      </div>
    </PageTransition>
  );
}
