"use client";
import { useEffect, useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Drawer, DrawerContent, DrawerDescription, DrawerHeader, DrawerTitle } from "@/components/ui/drawer";
import { linkCandidatesFor, linkContactToFriend, type LinkCandidateView } from "@/lib/actions/friends";

// After you become pals: if you already had a contact who could be them ("Pravin" for "Pravin Raj"), ask, so the
// friend isn't added twice and the number you saved stays. Resolves by calling `onDone` (also when there is nothing to ask).
export function LinkContactSheet({ friend, onDone }: { friend: { id: string; name: string } | null; onDone: () => void }) {
  const [found, setFound] = useState<LinkCandidateView[] | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  useEffect(() => {
    if (!friend) return;
    let cancelled = false;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- reset while the next friend's candidates load
    setFound(null);
    linkCandidatesFor(friend.id)
      .then((c) => !cancelled && (c.length === 0 ? onDone() : setFound(c)))
      .catch(() => !cancelled && onDone());
    return () => {
      cancelled = true;
    };
  }, [friend, onDone]);

  const use = async (c: LinkCandidateView) => {
    if (!friend) return;
    setBusy(c.id);
    try {
      await linkContactToFriend(friend.id, c.id);
      toast.success(`${c.name} is now ${friend.name}`);
      onDone();
    } catch {
      toast.error("Couldn't link that. Please try again.");
    } finally {
      setBusy(null);
    }
  };

  return (
    <Drawer open={!!friend && !!found && found.length > 0} onOpenChange={(o) => !o && onDone()}>
      <DrawerContent data-testid="link-contact">
        <DrawerHeader>
          <DrawerTitle className="text-foreground">Is this {friend?.name}?</DrawerTitle>
          <DrawerDescription className="text-muted-foreground">You already have someone saved who could be them. Link them to keep their name and number, and your past bills with them.</DrawerDescription>
        </DrawerHeader>
        <div className="mx-auto w-full max-w-md space-y-3 px-4 pb-6">
          <ul className="space-y-2">
            {(found ?? []).map((c) => (
              <li key={c.id} className="flex items-center gap-3 rounded-2xl glass-inset p-3">
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm font-bold">{c.name}</span>
                  <span className="block text-xs text-muted-foreground">{c.hasPhone ? "Has a WhatsApp number" : "No number saved"}</span>
                </span>
                <Button type="button" size="sm" data-testid="link-contact-yes" className="h-10 rounded-full px-4" disabled={busy === c.id} onClick={() => use(c)}>
                  Yes, that&apos;s them
                </Button>
              </li>
            ))}
          </ul>
          <Button type="button" variant="glass" className="h-12 w-full" onClick={onDone}>
            No, keep them separate
          </Button>
        </div>
      </DrawerContent>
    </Drawer>
  );
}
