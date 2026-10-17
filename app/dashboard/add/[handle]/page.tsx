"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { Check, Loader2, UserPlus } from "lucide-react";
import { toast } from "sonner";

import { PageTransition } from "@/components/page-transition";
import { PersonAvatar } from "@/components/person-avatar";
import { Button } from "@/components/ui/button";
import { lookupUser, sendFriendRequest, type LookupResult } from "@/lib/actions/friends";

// What an invite link or QR opens: "Add @alex". Signed-in only (see lib/auth.config.ts); the request still has to
// be accepted by the other person.
export default function AddFriendPage() {
  const params = useParams();
  const raw = decodeURIComponent(String(params?.handle ?? ""));
  const [found, setFound] = useState<LookupResult | null>(null);
  const [sending, setSending] = useState(false);
  const [done, setDone] = useState<"requested" | "accepted" | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    lookupUser(raw)
      .then((r) => !cancelled && setFound(r))
      .catch(() => !cancelled && setFound({ ok: false, error: "Couldn't load that. Check your connection." }));
    return () => {
      cancelled = true;
    };
  }, [raw]);

  const send = async () => {
    if (!found || !found.ok) return;
    setSending(true);
    setError(null);
    try {
      const r = await sendFriendRequest(found.user.username);
      if (r.ok) {
        setDone(r.status);
        toast.success(r.status === "accepted" ? "You're friends" : "Request sent");
      } else setError(r.error);
    } catch {
      setError("Couldn't send that. Please try again.");
    } finally {
      setSending(false);
    }
  };

  const name = found && found.ok ? (found.user.name ?? "").trim() || `@${found.user.username}` : "";

  return (
    <PageTransition>
      <div className="mx-auto mb-20 min-h-screen max-w-xl space-y-6 p-6 text-foreground md:p-10">
        <header className="space-y-1">
          <h1 className="type-title">Add a friend</h1>
          <p className="type-eyebrow">From an invite</p>
        </header>

        {found === null ? (
          <div className="h-32 animate-pulse rounded-3xl glass" aria-label="Loading" />
        ) : !found.ok ? (
          <div data-testid="add-not-found" className="space-y-3 rounded-3xl glass p-6 text-center">
            <p className="text-base font-semibold">{found.error}</p>
            <p className="text-sm text-muted-foreground">Check the link, or ask them for a fresh one.</p>
          </div>
        ) : (
          <div data-testid="add-card" className="space-y-5 rounded-3xl glass p-6 text-center">
            <PersonAvatar name={name} seed={found.user.avatarSeed} className="mx-auto size-20" />
            <div>
              <p className="text-xl font-bold">{name}</p>
              <p className="text-sm text-muted-foreground">@{found.user.username}</p>
            </div>
            {done ? (
              <p data-testid="add-done" className="flex items-center justify-center gap-2 text-sm font-semibold text-success">
                <Check size={16} strokeWidth={3} /> {done === "accepted" ? "You're friends" : `Request sent. ${name} needs to accept it.`}
              </p>
            ) : found.relation === "self" ? (
              <p className="text-sm text-muted-foreground">That&apos;s your own invite.</p>
            ) : found.relation === "friends" ? (
              <p className="flex items-center justify-center gap-2 text-sm font-semibold text-success"><Check size={16} strokeWidth={3} /> Already friends</p>
            ) : found.relation === "outgoing" ? (
              <p className="text-sm text-muted-foreground">You&apos;ve already asked. Waiting for {name}.</p>
            ) : (
              <Button type="button" data-testid="add-send" size="lg" className="w-full" disabled={sending} onClick={send}>
                {sending ? <Loader2 className="size-4 animate-spin" /> : <><UserPlus className="mr-2 size-4" /> {found.relation === "incoming" ? `Accept ${name}` : `Add ${name}`}</>}
              </Button>
            )}
            {error && (
              <p role="alert" className="text-sm text-coral">
                {error}
              </p>
            )}
          </div>
        )}

        <Link href="/dashboard/people" className="flex min-h-11 items-center justify-center text-sm font-semibold text-muted-foreground hover:text-foreground">
          Go to People
        </Link>
      </div>
    </PageTransition>
  );
}
