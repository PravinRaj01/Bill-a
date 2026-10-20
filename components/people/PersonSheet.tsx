"use client";
import { useState } from "react";
import Link from "next/link";
import { Loader2, MessageCircle, Trash2 } from "lucide-react";

import { PersonAvatar } from "@/components/person-avatar";
import { Button } from "@/components/ui/button";
import { Drawer, DrawerContent, DrawerDescription, DrawerHeader, DrawerTitle } from "@/components/ui/drawer";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import type { PersonView } from "@/lib/actions/people";
import { formatMoney } from "@/lib/money";
import { formatPhone } from "@/lib/people/phone";
import type { PersonBalance } from "@/lib/split/settle";

export interface RecentBill {
  id: string;
  title: string;
  createdAt: string | number | Date;
}

// Add a contact, or open one: what they owe, their latest bills, and the name and WhatsApp number to edit.
// The sheet is remounted per person (see the key at the call site), so its fields start from that person.
export function PersonSheet({
  person,
  balances = [],
  recent = [],
  open,
  onOpenChange,
  onSave,
  onDelete,
  onNudge,
  onSharePast,
  onUnfriend,
}: {
  /** null = a new contact. */
  person: PersonView | null;
  balances?: PersonBalance[];
  recent?: RecentBill[];
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Resolves to an error message, or null when saved. */
  onSave: (input: { name: string; phone: string }) => Promise<string | null>;
  onDelete?: () => void;
  onNudge?: (b: PersonBalance) => void;
  /** A pal on Bill.a: offer the bills you have already split with them. */
  onSharePast?: () => void;
  /** A pal on Bill.a can't be deleted while linked; ending the friendship is what removes the link. */
  onUnfriend?: () => void;
}) {
  const [name, setName] = useState(person?.name ?? "");
  const [phone, setPhone] = useState(person?.phone ? formatPhone(person.phone) : "");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const save = async () => {
    setError(null);
    setSaving(true);
    const message = await onSave({ name: name.trim(), phone: phone.trim() });
    setSaving(false);
    if (message) setError(message);
    else onOpenChange(false);
  };

  const owedToYou = balances.filter((b) => b.net > 0);

  return (
    <Drawer open={open} onOpenChange={onOpenChange}>
      <DrawerContent data-testid="person-sheet">
        <DrawerHeader>
          <DrawerTitle className="text-foreground">{person ? person.name : "Add a person"}</DrawerTitle>
          <DrawerDescription className="text-muted-foreground">
            {person?.friend ? `${person.friend.username ? `@${person.friend.username} · ` : ""}Pals on Bill.a` : person ? "Their number lets a nudge open their WhatsApp chat." : "Save a name, and a WhatsApp number if you have it."}
          </DrawerDescription>
        </DrawerHeader>

        <div className="mx-auto max-h-[72vh] w-full max-w-md space-y-5 overflow-y-auto px-4 pb-6" data-vaul-no-drag>
          {person && (
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <PersonAvatar name={person.name} seed={person.friend?.avatarSeed} className="size-12" />
                <div className="min-w-0 flex-1" data-testid="person-balance">
                  {balances.length === 0 ? (
                    <p className="text-sm text-muted-foreground">All square. Nothing owed either way.</p>
                  ) : (
                    balances.map((b) => (
                      <p key={b.key} className="text-sm text-muted-foreground">
                        {b.net > 0 ? "Owes you " : "You owe "}
                        <strong className={`font-bold tabular-nums ${b.net > 0 ? "text-success" : "text-coral"}`}>{formatMoney(Math.abs(b.net), b.currency)}</strong>
                      </p>
                    ))
                  )}
                </div>
              </div>
              {owedToYou.length > 0 && person.phone && onNudge && (
                <Button type="button" variant="whatsapp" size="lg" data-testid="person-nudge" className="w-full" onClick={() => onNudge(owedToYou[0])}>
                  <MessageCircle className="mr-2 size-4" /> Nudge on WhatsApp
                </Button>
              )}
              {recent.length > 0 && (
                <div className="space-y-1.5">
                  <p className="type-eyebrow">Recent bills</p>
                  <ul className="space-y-1">
                    {recent.map((b) => (
                      <li key={b.id}>
                        <Link
                          href={`/dashboard/history/${b.id}`}
                          data-testid="person-bill"
                          className="flex min-h-11 items-center justify-between gap-3 rounded-xl px-3 text-sm transition-colors hover:bg-white/5"
                        >
                          <span className="truncate font-semibold">{b.title}</span>
                          <span className="shrink-0 text-xs text-muted-foreground">{new Date(b.createdAt).toLocaleDateString(undefined, { day: "numeric", month: "short" })}</span>
                        </Link>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          <div className="space-y-4">
            <div className="space-y-1.5">
              <Label htmlFor="person-name" className="text-xs font-semibold text-muted-foreground">
                Name
              </Label>
              <Input id="person-name" data-testid="person-name" maxLength={60} value={name} onChange={(e) => setName(e.target.value)} autoComplete="off" />
              {person && <p className="text-xs text-muted-foreground">Renaming doesn&apos;t change bills you&apos;ve already saved.</p>}
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="person-phone" className="text-xs font-semibold text-muted-foreground">
                WhatsApp number <span className="font-normal">(optional)</span>
              </Label>
              <Input
                id="person-phone"
                data-testid="person-phone"
                type="tel"
                inputMode="tel"
                autoComplete="off"
                placeholder="012-345 6789"
                maxLength={40}
                value={phone}
                onChange={(e) => setPhone(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && name.trim() && (e.preventDefault(), void save())}
              />
            </div>
          </div>

          {error && (
            <p role="alert" data-testid="person-error" className="text-sm text-coral">
              {error}
            </p>
          )}

          {person?.linkedUserId && (onSharePast || onUnfriend) && (
            <div className="flex gap-3">
              {onSharePast && (
                <Button type="button" variant="glass" data-testid="friend-share-past" className="h-11 flex-1" onClick={onSharePast}>
                  Share past bills
                </Button>
              )}
              {onUnfriend && (
                <Button type="button" variant="outline" data-testid="friend-remove" className="h-11 flex-1 rounded-xl border-white/10 text-destructive hover:bg-destructive/10 hover:text-destructive" onClick={onUnfriend}>
                  Unfriend
                </Button>
              )}
            </div>
          )}

          <div className="flex gap-3">
            <Button type="button" data-testid="person-save" onClick={save} disabled={!name.trim() || saving} className="h-12 flex-1">
              {saving && <Loader2 className="mr-2 size-4 animate-spin" />} {person ? "Save" : "Add"}
            </Button>
            {person && onDelete && !person.linkedUserId && (
              <Button type="button" data-testid="person-delete" variant="outline" onClick={onDelete} aria-label="Delete person" className="h-12 rounded-xl border-white/10 px-4 text-destructive hover:bg-destructive/10 hover:text-destructive">
                <Trash2 size={18} />
              </Button>
            )}
          </div>
        </div>
      </DrawerContent>
    </Drawer>
  );
}
