"use client";
import { X } from "lucide-react";

import { PersonAvatar } from "@/components/person-avatar";
import { Button } from "@/components/ui/button";
import type { FriendView, FriendsView } from "@/lib/actions/friends";

export const friendLabel = (u: { name: string | null; username: string | null }) => (u.name ?? "").trim() || (u.username ? `@${u.username}` : "A friend");

// Requests sit above the list: someone waiting on you comes first, and a request you sent can be taken back.
export function FriendRequests({
  state,
  onAnswer,
  onCancel,
}: {
  state: FriendsView | null;
  onAnswer: (u: FriendView, accept: boolean) => void;
  onCancel: (u: FriendView) => void;
}) {
  if (!state || (state.incoming.length === 0 && state.outgoing.length === 0)) return null;
  return (
    <section aria-label="Friend requests" data-testid="friend-requests" className="space-y-2">
      <h2 className="type-eyebrow px-1">Requests</h2>
      <ul className="space-y-2">
        {state.incoming.map((u) => (
          <li key={u.id} data-testid="request-incoming" className="flex items-center gap-3 rounded-3xl glass p-3">
            <PersonAvatar name={friendLabel(u)} seed={u.avatarSeed} className="size-11" />
            <span className="min-w-0 flex-1">
              <span className="block truncate text-sm font-bold">{friendLabel(u)}</span>
              <span className="block truncate text-xs text-muted-foreground">wants to be pals{u.username ? ` · @${u.username}` : ""}</span>
            </span>
            <Button type="button" size="sm" data-testid="request-accept" className="h-10 rounded-full px-4" onClick={() => onAnswer(u, true)}>
              Accept
            </Button>
            <button type="button" aria-label={`Decline ${friendLabel(u)}`} data-testid="request-decline" onClick={() => onAnswer(u, false)} className="flex size-10 items-center justify-center rounded-full text-muted-foreground hover:bg-white/10">
              <X size={16} />
            </button>
          </li>
        ))}
        {state.outgoing.map((u) => (
          <li key={u.id} data-testid="request-outgoing" className="flex items-center gap-3 rounded-3xl glass-inset p-3">
            <PersonAvatar name={friendLabel(u)} seed={u.avatarSeed} className="size-10" />
            <span className="min-w-0 flex-1">
              <span className="block truncate text-sm font-semibold">{friendLabel(u)}</span>
              <span className="block truncate text-xs text-muted-foreground">Requested{u.username ? ` · @${u.username}` : ""}</span>
            </span>
            <button type="button" data-testid="request-cancel" onClick={() => onCancel(u)} className="min-h-10 rounded-full px-3 text-xs font-semibold text-muted-foreground hover:bg-white/10 hover:text-foreground">
              Cancel
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
