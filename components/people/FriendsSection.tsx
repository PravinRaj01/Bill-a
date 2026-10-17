"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Check, Copy, Loader2, QrCode, Search, UserPlus, X } from "lucide-react";
import { QRCodeSVG } from "qrcode.react";
import { toast } from "sonner";

import { PastBillsSheet } from "@/components/people/PastBillsSheet";
import { PersonAvatar } from "@/components/person-avatar";
import { useProfile } from "@/components/profile-provider";
import { Button } from "@/components/ui/button";
import { ConfirmSheet } from "@/components/ui/confirm-sheet";
import { Input } from "@/components/ui/input";
import { getFriends, lookupUser, respondToRequest, sendFriendRequest, unfriend, type FriendView, type FriendsView, type LookupResult } from "@/lib/actions/friends";
import { shareText, shareMessage } from "@/lib/ui/share";

const label = (u: { name: string | null; username: string | null }) => (u.name ?? "").trim() || (u.username ? `@${u.username}` : "A friend");

// Friends on Bill.a: find someone by their exact @username (or hand them your invite link / QR), answer requests,
// and see who you're linked with. Linked friends' bills show up in each other's History.
export function FriendsSection({ onChanged, focusSearch = false }: { onChanged: () => void; focusSearch?: boolean }) {
  const handleRef = useRef<HTMLInputElement>(null);
  const profile = useProfile();
  const [state, setState] = useState<FriendsView | null>(null);
  const [handle, setHandle] = useState("");
  const [found, setFound] = useState<LookupResult | null>(null);
  const [searching, setSearching] = useState(false);
  const [sending, setSending] = useState(false);
  const [showQr, setShowQr] = useState(false);
  const [offer, setOffer] = useState<{ id: string; name: string } | null>(null);
  const [removing, setRemoving] = useState<FriendView | null>(null);

  const reload = useCallback(async () => {
    try {
      setState(await getFriends());
    } catch {
      setState((s) => s ?? { friends: [], incoming: [], outgoing: [] });
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- load from the server on mount
    void reload();
  }, [reload]);

  // Arriving from "Add a friend" in the + sheet: the @username box is ready to type in.
  useEffect(() => {
    if (!focusSearch) return;
    const t = setTimeout(() => handleRef.current?.focus({ preventScroll: false }), 350);
    return () => clearTimeout(t);
  }, [focusSearch]);

  const inviteUrl = profile.username && typeof window !== "undefined" ? `${window.location.origin}/dashboard/add/${profile.username}` : null;

  const search = async () => {
    const h = handle.trim();
    if (!h) return;
    setSearching(true);
    setFound(null);
    try {
      setFound(await lookupUser(h));
    } catch {
      setFound({ ok: false, error: "Couldn't search. Check your connection." });
    } finally {
      setSearching(false);
    }
  };

  const send = async () => {
    if (!found || !found.ok) return;
    setSending(true);
    try {
      const r = await sendFriendRequest(found.user.username);
      if (!r.ok) {
        setFound({ ok: false, error: r.error });
        return;
      }
      toast.success(r.status === "accepted" ? `You and ${label(found.user)} are friends` : "Request sent");
      setHandle("");
      setFound(null);
      await reload();
      if (r.status === "accepted") onChanged();
    } catch {
      toast.error("Couldn't send that. Please try again.");
    } finally {
      setSending(false);
    }
  };

  const answer = async (u: FriendView, accept: boolean) => {
    try {
      await respondToRequest(u.id, accept);
      await reload();
      if (accept) {
        onChanged();
        setOffer({ id: u.id, name: label(u) });
      }
    } catch {
      toast.error("Couldn't answer that. Please try again.");
    }
  };

  const remove = async (u: FriendView) => {
    try {
      await unfriend(u.id);
      await reload();
      onChanged();
      toast("Unfriended. Bills you shared are no longer visible to each other.");
    } catch {
      toast.error("Couldn't do that. Please try again.");
    }
  };

  const copyInvite = async () => {
    if (!inviteUrl) return;
    const msg = shareMessage(await shareText(`Add me on Bill.a: ${inviteUrl}`));
    if (msg) toast(msg);
  };

  return (
    <section aria-label="Friends" data-testid="friends-section" className="space-y-4">
      <h2 className="type-eyebrow">Friends on Bill.a</h2>

      {/* Incoming requests come first: someone is waiting on you. */}
      {state && state.incoming.length > 0 && (
        <ul className="space-y-2" data-testid="friend-requests">
          {state.incoming.map((u) => (
            <li key={u.id} className="flex items-center gap-3 rounded-3xl glass p-3">
              <PersonAvatar name={label(u)} seed={u.avatarSeed} className="size-11" />
              <span className="min-w-0 flex-1">
                <span className="block truncate text-sm font-bold">{label(u)}</span>
                <span className="block truncate text-xs text-muted-foreground">wants to be friends{u.username ? ` · @${u.username}` : ""}</span>
              </span>
              <Button type="button" size="sm" data-testid="request-accept" className="h-10 rounded-full px-4" onClick={() => answer(u, true)}>
                Accept
              </Button>
              <button type="button" aria-label={`Decline ${label(u)}`} data-testid="request-decline" onClick={() => answer(u, false)} className="flex size-10 items-center justify-center rounded-full text-muted-foreground hover:bg-white/10">
                <X size={16} />
              </button>
            </li>
          ))}
        </ul>
      )}

      {/* Find someone by their exact @username. */}
      <div className="space-y-2 rounded-3xl glass p-4">
        <div className="flex gap-2">
          <Input
            ref={handleRef}
            aria-label="Friend's @username"
            data-testid="friend-handle"
            placeholder="@username"
            autoCapitalize="none"
            autoCorrect="off"
            spellCheck={false}
            maxLength={21}
            value={handle}
            onChange={(e) => {
              setHandle(e.target.value);
              setFound(null);
            }}
            onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), void search())}
          />
          <Button type="button" data-testid="friend-search" variant="glass" className="size-12 shrink-0 p-0" aria-label="Find friend" disabled={!handle.trim() || searching} onClick={search}>
            {searching ? <Loader2 className="size-4 animate-spin" /> : <Search size={18} />}
          </Button>
        </div>
        {found && !found.ok && (
          <p role="alert" data-testid="friend-error" className="px-1 text-sm text-coral">
            {found.error}
          </p>
        )}
        {found && found.ok && (
          <div data-testid="friend-found" className="flex items-center gap-3 rounded-2xl glass-inset p-3">
            <PersonAvatar name={label(found.user)} seed={found.user.avatarSeed} className="size-11" />
            <span className="min-w-0 flex-1">
              <span className="block truncate text-sm font-bold">{label(found.user)}</span>
              <span className="block truncate text-xs text-muted-foreground">@{found.user.username}</span>
            </span>
            {found.relation === "none" && (
              <Button type="button" data-testid="friend-send" size="sm" className="h-10 rounded-full px-4" disabled={sending} onClick={send}>
                {sending ? <Loader2 className="size-4 animate-spin" /> : <><UserPlus className="mr-1 size-4" /> Add</>}
              </Button>
            )}
            {found.relation === "friends" && <span className="flex items-center gap-1 text-xs font-semibold text-success"><Check size={14} /> Friends</span>}
            {found.relation === "outgoing" && <span className="text-xs font-semibold text-muted-foreground">Requested</span>}
            {found.relation === "incoming" && <span className="text-xs font-semibold text-muted-foreground">Asked you: see above</span>}
            {found.relation === "self" && <span className="text-xs font-semibold text-muted-foreground">That&apos;s you</span>}
          </div>
        )}
      </div>

      {/* Friends already linked. */}
      {state && state.friends.length > 0 && (
        <ul className="space-y-2" data-testid="friend-list">
          {state.friends.map((u) => (
            <li key={u.id} className="flex items-center gap-3 rounded-3xl glass p-3">
              <PersonAvatar name={label(u)} seed={u.avatarSeed} className="size-11" />
              <span className="min-w-0 flex-1">
                <span className="block truncate text-sm font-bold">{label(u)}</span>
                <span className="block truncate text-xs text-muted-foreground">{u.username ? `@${u.username} · ` : ""}Friends</span>
              </span>
              <Button type="button" size="sm" variant="glass" data-testid="friend-share-past" className="h-10 rounded-full px-3 text-xs" onClick={() => setOffer({ id: u.id, name: label(u) })}>
                Share past bills
              </Button>
              <button type="button" aria-label={`Unfriend ${label(u)}`} data-testid="friend-remove" onClick={() => setRemoving(u)} className="flex size-10 items-center justify-center rounded-full text-muted-foreground hover:bg-white/10 hover:text-destructive">
                <X size={16} />
              </button>
            </li>
          ))}
        </ul>
      )}
      {state && state.outgoing.length > 0 && (
        <p data-testid="friend-pending" className="px-1 text-xs text-muted-foreground">
          Waiting on {state.outgoing.map(label).join(", ")}
        </p>
      )}

      {/* Your own invite: link and QR. Needs a username. */}
      <div className="space-y-3 rounded-3xl glass p-4">
        {profile.username ? (
          <>
            <div className="flex items-center gap-3">
              <span className="min-w-0 flex-1">
                <span className="block text-sm font-bold">Your invite</span>
                <span className="block truncate text-xs text-muted-foreground">@{profile.username}. Anyone with this link can ask to be your friend.</span>
              </span>
              <Button type="button" variant="glass" size="sm" data-testid="invite-copy" className="h-10 rounded-full px-3 text-xs" onClick={copyInvite}>
                <Copy className="mr-1 size-3.5" /> Share
              </Button>
              <Button type="button" variant="glass" size="sm" data-testid="invite-qr-toggle" aria-expanded={showQr} className="h-10 rounded-full px-3 text-xs" onClick={() => setShowQr((v) => !v)}>
                <QrCode className="mr-1 size-3.5" /> QR
              </Button>
            </div>
            {showQr && inviteUrl && (
              <div className="flex justify-center">
                <div className="rounded-2xl bg-white p-3" data-testid="invite-qr">
                  <QRCodeSVG value={inviteUrl} size={168} level="M" marginSize={1} title="Your Bill.a invite" />
                </div>
              </div>
            )}
          </>
        ) : (
          <p className="text-sm text-muted-foreground">
            Pick a username to get an invite link and let friends find you.{" "}
            <Link href="/dashboard/account?tab=profile" className="font-semibold text-foreground underline underline-offset-4">
              Choose one
            </Link>
          </p>
        )}
      </div>

      <PastBillsSheet friend={offer} onClose={() => setOffer(null)} />
      <ConfirmSheet
        open={!!removing}
        onOpenChange={(o) => !o && setRemoving(null)}
        title={`Unfriend ${removing ? label(removing) : "this person"}?`}
        description="You stop seeing each other's shared bills right away. Your contact for them stays as a plain name."
        confirmLabel="Unfriend"
        onConfirm={async () => {
          if (removing) await remove(removing);
        }}
      />
    </section>
  );
}
