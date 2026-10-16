"use client";
import { useEffect, useState } from "react";

import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { avatarDataUri, cachedAvatar } from "@/lib/ui/avatar";
import { cn } from "@/lib/utils";
import { initials, personColor } from "@/lib/ui/person-color";

/** The person's portrait: instantly if it has been made before, else null until it arrives. */
function usePortrait(name: string): string | null {
  const [made, setMade] = useState<{ name: string; uri: string } | null>(null);
  useEffect(() => {
    if (cachedAvatar(name)) return;
    let cancelled = false;
    avatarDataUri(name)
      .then((uri) => !cancelled && setMade({ name, uri }))
      .catch(() => {
        /* the initials stay */
      });
    return () => {
      cancelled = true;
    };
  }, [name]);
  return cachedAvatar(name) ?? (made?.name === name ? made.uri : null);
}

// A hand-drawn portrait made from the name (lib/ui/avatar.ts) inside the person's stable colour ring. The
// initials show while the portrait loads, and stay if it can't.
export function PersonAvatar({ name, className, compact }: { name: string; className?: string; compact?: boolean }) {
  const portrait = usePortrait(name);
  return (
    <Avatar
      className={cn("size-9", compact ? "border-2" : "ring-2", className)}
      style={compact ? { borderColor: personColor(name) } : { ["--tw-ring-color" as string]: personColor(name) }}
      title={name}
    >
      {portrait && <AvatarImage src={portrait} alt="" />}
      <AvatarFallback className={cn("bg-surface-2 text-xs font-bold text-foreground", compact && "text-[10px]")}>{initials(name)}</AvatarFallback>
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
        <span className="flex size-8 items-center justify-center rounded-full border-2 border-surface bg-surface-2 text-[11px] font-bold">
          +{extra}
        </span>
      )}
    </div>
  );
}
