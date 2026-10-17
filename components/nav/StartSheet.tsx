"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { Calculator, Camera, ChevronRight, Users } from "lucide-react";

import { AvatarStack } from "@/components/person-avatar";
import { Drawer, DrawerContent, DrawerDescription, DrawerHeader, DrawerTitle } from "@/components/ui/drawer";
import { listGroups } from "@/lib/actions/groups";
import type { SavedGroup } from "@/lib/db/schema";

const tile =
  "flex min-h-28 flex-col justify-between rounded-3xl border border-white/10 bg-surface-2 p-4 text-left outline-none transition-colors hover:border-white/25 focus-visible:ring-2 focus-visible:ring-white/30 active:scale-[0.98]";

// "Start a split" from anywhere (the + in the bottom nav and the sidebar button): scan a receipt, type an
// amount, or begin with a saved group. Groups are an account feature, so guests just see the two tiles.
export function StartSheet({ open, onOpenChange, signedIn }: { open: boolean; onOpenChange: (open: boolean) => void; signedIn: boolean }) {
  const [groups, setGroups] = useState<SavedGroup[]>([]);

  useEffect(() => {
    if (!open || !signedIn) return;
    let cancelled = false;
    listGroups()
      .then((g) => !cancelled && setGroups(g))
      .catch(() => !cancelled && setGroups([]));
    return () => {
      cancelled = true;
    };
  }, [open, signedIn]);

  const close = () => onOpenChange(false);

  return (
    <Drawer open={open} onOpenChange={onOpenChange}>
      <DrawerContent data-testid="start-sheet">
        <DrawerHeader>
          <DrawerTitle className="text-white">Start a split</DrawerTitle>
          <DrawerDescription className="text-muted-foreground">How do you want to begin?</DrawerDescription>
        </DrawerHeader>

        <div className="mx-auto w-full max-w-md space-y-5 px-4 pb-8">
          <div className="grid grid-cols-2 gap-3">
            <Link href="/dashboard/new" transitionTypes={["nav-forward"]} onClick={close} data-testid="start-scan" className={tile}>
              <span className="flex size-11 items-center justify-center rounded-full bg-coral/15 text-coral">
                <Camera className="size-5" />
              </span>
              <span>
                <span className="block text-base font-bold">Scan receipt</span>
                <span className="block text-xs text-muted-foreground">We read the items</span>
              </span>
            </Link>
            <Link href="/dashboard/new?mode=quick" transitionTypes={["nav-forward"]} onClick={close} data-testid="start-quick" className={tile}>
              <span className="flex size-11 items-center justify-center rounded-full bg-white/10">
                <Calculator className="size-5" />
              </span>
              <span>
                <span className="block text-base font-bold">Quick split</span>
                <span className="block text-xs text-muted-foreground">Just an amount</span>
              </span>
            </Link>
          </div>

          {signedIn && groups.length > 0 && (
            <section aria-label="With a saved group" className="space-y-2">
              <div className="flex items-center justify-between px-1">
                <h3 className="flex items-center gap-2 text-xs font-semibold tracking-widest text-muted-foreground uppercase">
                  <Users size={14} /> With a group
                </h3>
                <Link href="/dashboard/groups" transitionTypes={["nav-forward"]} onClick={close} className="text-xs font-semibold text-muted-foreground hover:text-white">
                  All groups
                </Link>
              </div>
              <ul className="space-y-2">
                {groups.slice(0, 4).map((g) => (
                  <li key={g.id}>
                    <Link
                      href={`/dashboard/new?group_id=${g.id}`}
                      transitionTypes={["nav-forward"]}
                      onClick={close}
                      data-testid="start-group"
                      className="flex min-h-16 items-center gap-3 rounded-2xl bg-surface-2 px-4 py-2 transition-colors hover:bg-white/10"
                    >
                      <AvatarStack names={g.names} max={3} />
                      <span className="min-w-0 flex-1">
                        <span className="block truncate text-sm font-bold">{g.groupName}</span>
                        <span className="block truncate text-xs text-muted-foreground">{g.names.join(", ")}</span>
                      </span>
                      <ChevronRight size={16} className="shrink-0 text-muted-foreground" />
                    </Link>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </div>
      </DrawerContent>
    </Drawer>
  );
}
