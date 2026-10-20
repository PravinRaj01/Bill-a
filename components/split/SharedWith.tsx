"use client";
import { useState } from "react";
import { Loader2, Share2 } from "lucide-react";

import { PersonAvatar } from "@/components/person-avatar";
import type { PersonView } from "@/lib/actions/people";
import type { Participant } from "@/types/domain";

const key = (s: string) => s.trim().toLowerCase();

// Who a bill is shared with. A friend on the bill sees it in their History, so this says so plainly and lets the
// owner stop (or start) sharing per person. Stopping keeps the person on the bill as a plain name.
export function SharedWith({
  participants,
  contacts,
  onChange,
}: {
  participants: Participant[] | null;
  contacts: PersonView[] | null;
  onChange: (name: string, userId: string | null) => void | Promise<void>;
}) {
  const [busy, setBusy] = useState<string | null>(null);
  if (!participants || !contacts) return null;

  const byAccount = new Map(contacts.filter((c) => c.linkedUserId).map((c) => [c.linkedUserId as string, c]));
  const byName = new Map(contacts.filter((c) => c.linkedUserId).map((c) => [key(c.name), c]));
  const shared = participants.filter((p) => p.userId && !p.me);
  const offers = participants.flatMap((p) => {
    const c = !p.userId && !p.me ? byName.get(key(p.name)) : undefined;
    return c ? [{ p, c }] : [];
  });
  if (shared.length === 0 && offers.length === 0) return null;

  const run = async (name: string, userId: string | null) => {
    setBusy(name);
    try {
      await onChange(name, userId);
    } finally {
      setBusy(null);
    }
  };

  return (
    <section aria-label="Shared with" data-testid="shared-with" className="space-y-2">
      <h2 className="type-eyebrow px-1">Shared with</h2>
      <ul className="space-y-2">
        {shared.map((p) => {
          const c = byAccount.get(p.userId as string);
          const handle = c?.friend?.username;
          return (
            <li key={p.name} data-testid="shared-with-row" className="flex items-center gap-3 rounded-2xl glass p-3">
              <PersonAvatar name={p.name} seed={c?.friend?.avatarSeed} className="size-10" />
              <span className="min-w-0 flex-1">
                <span className="block truncate text-sm font-bold">{p.name}</span>
                <span className="block truncate text-xs text-muted-foreground">{handle ? `@${handle} · ` : ""}in their History</span>
              </span>
              <button
                type="button"
                data-testid="unshare"
                disabled={busy === p.name}
                onClick={() => void run(p.name, null)}
                className="min-h-10 rounded-full px-3 text-xs font-semibold text-muted-foreground transition-colors hover:bg-white/10 hover:text-foreground disabled:opacity-50"
              >
                {busy === p.name ? <Loader2 className="size-4 animate-spin" /> : "Unshare"}
              </button>
            </li>
          );
        })}
        {offers.map(({ p, c }) => (
          <li key={p.name} data-testid="share-offer" className="flex items-center gap-3 rounded-2xl glass-inset p-3">
            <PersonAvatar name={p.name} seed={c.friend?.avatarSeed} className="size-10" />
            <span className="min-w-0 flex-1">
              <span className="block truncate text-sm font-semibold">{p.name}</span>
              <span className="block truncate text-xs text-muted-foreground">Not shared with them</span>
            </span>
            <button
              type="button"
              data-testid="share-with"
              disabled={busy === p.name}
              onClick={() => void run(p.name, c.linkedUserId as string)}
              className="flex min-h-10 items-center gap-1 rounded-full bg-white/10 px-3 text-xs font-semibold text-foreground transition-colors hover:bg-white/15 disabled:opacity-50"
            >
              {busy === p.name ? <Loader2 className="size-4 animate-spin" /> : <Share2 className="size-3.5" />} Share
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
