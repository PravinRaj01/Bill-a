"use client";
import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Check, Copy, Loader2, Search, UserPlus } from "lucide-react";
import { QRCodeSVG } from "qrcode.react";
import { toast } from "sonner";

import { friendLabel } from "@/components/people/FriendRequests";
import { PersonAvatar } from "@/components/person-avatar";
import { useProfile } from "@/components/profile-provider";
import { Button } from "@/components/ui/button";
import { Drawer, DrawerContent, DrawerDescription, DrawerHeader, DrawerTitle } from "@/components/ui/drawer";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Segmented } from "@/components/ui/segmented";
import { lookupUser, sendFriendRequest, type LookupResult } from "@/lib/actions/friends";
import { shareMessage, shareText } from "@/lib/ui/share";

export type AddTab = "friend" | "name" | "invite";

// One place to add a pal: find a friend by @username, save just a name (and a WhatsApp number), or hand over your own
// invite link and QR. The sheet is remounted each time it opens (see the key at the call site).
export function AddPalSheet({
  open,
  onOpenChange,
  initialTab = "friend",
  onSent,
  onAddName,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  initialTab?: AddTab;
  /** A request went out, or a crossing request made you friends at once. */
  onSent: (status: "requested" | "accepted", username: string) => void;
  /** Resolves to an error message, or null when saved. */
  onAddName: (input: { name: string; phone: string }) => Promise<string | null>;
}) {
  const profile = useProfile();
  const [tab, setTab] = useState<AddTab>(initialTab);
  const handleRef = useRef<HTMLInputElement>(null);
  const [handle, setHandle] = useState("");
  const [found, setFound] = useState<LookupResult | null>(null);
  const [searching, setSearching] = useState(false);
  const [sending, setSending] = useState(false);
  const [name, setName] = useState("");
  const [phone, setPhone] = useState("");
  const [nameError, setNameError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [showQr, setShowQr] = useState(false);

  useEffect(() => {
    if (!open || tab !== "friend") return;
    const t = setTimeout(() => handleRef.current?.focus(), 350);
    return () => clearTimeout(t);
  }, [open, tab]);

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
      toast.success(r.status === "accepted" ? `You and ${friendLabel(found.user)} are pals` : "Request sent");
      onOpenChange(false);
      onSent(r.status, found.user.username);
    } catch {
      toast.error("Couldn't send that. Please try again.");
    } finally {
      setSending(false);
    }
  };

  const addName = async () => {
    setNameError(null);
    setSaving(true);
    const message = await onAddName({ name: name.trim(), phone: phone.trim() });
    setSaving(false);
    if (message) setNameError(message);
    else onOpenChange(false);
  };

  const copyInvite = async () => {
    if (!inviteUrl) return;
    const msg = shareMessage(await shareText(`Add me on Bill.a: ${inviteUrl}`));
    if (msg) toast(msg);
  };

  return (
    <Drawer open={open} onOpenChange={onOpenChange}>
      <DrawerContent data-testid="add-pal">
        <DrawerHeader>
          <DrawerTitle className="text-foreground">Add a pal</DrawerTitle>
          <DrawerDescription className="text-muted-foreground">Find a friend on Bill.a, or just save a name.</DrawerDescription>
        </DrawerHeader>

        <div className="mx-auto max-h-[72vh] w-full max-w-md space-y-5 overflow-y-auto px-4 pb-6" data-vaul-no-drag>
          <Segmented
            label="How to add"
            value={tab}
            onValueChange={setTab}
            options={[
              { value: "friend", label: "@username" },
              { value: "name", label: "Just a name" },
              { value: "invite", label: "Invite" },
            ]}
          />

          {tab === "friend" && (
            <div className="space-y-3">
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
                  <PersonAvatar name={friendLabel(found.user)} seed={found.user.avatarSeed} className="size-11" />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm font-bold">{friendLabel(found.user)}</span>
                    <span className="block truncate text-xs text-muted-foreground">@{found.user.username}</span>
                  </span>
                  {found.relation === "none" && (
                    <Button type="button" data-testid="friend-send" size="sm" className="h-10 rounded-full px-4" disabled={sending} onClick={send}>
                      {sending ? <Loader2 className="size-4 animate-spin" /> : <><UserPlus className="mr-1 size-4" /> Add</>}
                    </Button>
                  )}
                  {found.relation === "friends" && <span className="flex items-center gap-1 text-xs font-semibold text-success"><Check size={14} /> Pals</span>}
                  {found.relation === "outgoing" && <span className="text-xs font-semibold text-muted-foreground">Requested</span>}
                  {found.relation === "incoming" && <span className="text-xs font-semibold text-muted-foreground">Asked you: see Requests</span>}
                  {found.relation === "self" && <span className="text-xs font-semibold text-muted-foreground">That&apos;s you</span>}
                </div>
              )}
              <p className="px-1 text-xs text-muted-foreground">They have to accept. Once they do, bills you split with them show up in their History.</p>
            </div>
          )}

          {tab === "name" && (
            <div className="space-y-4">
              <div className="space-y-1.5">
                <Label htmlFor="pal-name" className="text-xs font-semibold text-muted-foreground">Name</Label>
                <Input id="pal-name" data-testid="person-name" maxLength={60} value={name} onChange={(e) => setName(e.target.value)} autoComplete="off" />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="pal-phone" className="text-xs font-semibold text-muted-foreground">
                  WhatsApp number <span className="font-normal">(optional)</span>
                </Label>
                <Input
                  id="pal-phone"
                  data-testid="person-phone"
                  type="tel"
                  inputMode="tel"
                  autoComplete="off"
                  placeholder="012-345 6789"
                  maxLength={40}
                  value={phone}
                  onChange={(e) => setPhone(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && name.trim() && (e.preventDefault(), void addName())}
                />
              </div>
              {nameError && (
                <p role="alert" data-testid="person-error" className="text-sm text-coral">{nameError}</p>
              )}
              <Button type="button" data-testid="person-save" onClick={addName} disabled={!name.trim() || saving} className="h-12 w-full">
                {saving && <Loader2 className="mr-2 size-4 animate-spin" />} Add
              </Button>
            </div>
          )}

          {tab === "invite" && (
            <div className="space-y-3">
              {profile.username ? (
                <>
                  <p className="px-1 text-sm text-muted-foreground">@{profile.username}. Anyone with this link can ask to be your pal.</p>
                  <div className="flex gap-2">
                    <Button type="button" variant="glass" data-testid="invite-copy" className="h-12 flex-1" onClick={copyInvite}>
                      <Copy className="mr-2 size-4" /> Share link
                    </Button>
                    <Button type="button" variant="glass" data-testid="invite-qr-toggle" aria-expanded={showQr} className="h-12 flex-1" onClick={() => setShowQr((v) => !v)}>
                      {showQr ? "Hide QR" : "Show QR"}
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
                  <Link href="/dashboard/account?tab=profile" className="font-semibold text-foreground underline underline-offset-4">Choose one</Link>
                </p>
              )}
            </div>
          )}
        </div>
      </DrawerContent>
    </Drawer>
  );
}
