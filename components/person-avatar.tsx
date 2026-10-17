"use client";
import { useEffect, useState, useSyncExternalStore } from "react";

import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { useSeedFor } from "@/components/profile-provider";
import { avatarDataUri, cachedAvatar } from "@/lib/ui/avatar";
import { cn } from "@/lib/utils";
import { initials, personColor } from "@/lib/ui/person-color";

// False on the server and while hydrating, true afterwards (and straight away on client navigations). The portrait
// cache only exists in the browser, so reading it during hydration would render a picture where the server drew
// initials: a hydration mismatch.
const noopSubscribe = () => () => {};
const useIsClient = () => useSyncExternalStore(noopSubscribe, () => true, () => false);

/** The person's portrait: instantly if it has been made before, else null until it arrives. */
function usePortrait(name: string, seed?: string): string | null {
  const [made, setMade] = useState<{ key: string; uri: string } | null>(null);
  const key = `${name}|${seed ?? ""}`;
  const isClient = useIsClient();
  useEffect(() => {
    if (cachedAvatar(name, seed)) return;
    let cancelled = false;
    avatarDataUri(name, seed)
      .then((uri) => !cancelled && setMade({ key, uri }))
      .catch(() => {
        /* the initials stay */
      });
    return () => {
      cancelled = true;
    };
  }, [name, seed, key]);
  if (!isClient) return null;
  return cachedAvatar(name, seed) ?? (made?.key === key ? made.uri : null);
}

// A hand-drawn portrait made from the name (lib/ui/avatar.ts) inside the person's stable colour ring. The
// initials show while the portrait loads, and stay if it can't.
export function PersonAvatar({ name, className, compact, seed }: { name: string; className?: string; compact?: boolean; seed?: string | null }) {
  // The user's own chosen portrait (Account › Profile) wins over the one generated from their name.
  // A friend's own chosen portrait (`seed`) is shown as they picked it.
  const own = useSeedFor(name);
  const portrait = usePortrait(name, seed ?? own);
  return (
    <Avatar
      className={cn("size-9", compact ? "border-2" : "ring-2", className)}
      style={compact ? { borderColor: personColor(name) } : { ["--tw-ring-color" as string]: personColor(name) }}
      title={name}
    >
      {portrait && <AvatarImage src={portrait} alt="" />}
      <AvatarFallback className={cn("bg-white/10 text-xs font-bold text-foreground", compact && "text-[10px]")}>{initials(name)}</AvatarFallback>
    </Avatar>
  );
}

/** Overlapping stack for "Splitting with …". */
export function AvatarStack({ names, max = 4 }: { names: string[]; max?: number }) {
  const shown = names.slice(0, max);
  const extra = names.length - shown.length;
  return (
    <div className="flex -space-x-1.5" aria-label={`Splitting with ${names.join(", ")}`}>
      {shown.map((n) => (
        <PersonAvatar key={n} name={n} compact className="size-8" />
      ))}
      {extra > 0 && (
        <span className="flex size-8 items-center justify-center rounded-full border-2 border-surface bg-white/10 text-[11px] font-bold">
          +{extra}
        </span>
      )}
    </div>
  );
}
