import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { cn } from "@/lib/utils";
import { initials, personColor } from "@/lib/ui/person-color";

// Initials in the person's stable colour. Illustrated avatars come later (Batch 4).
export function PersonAvatar({ name, className, compact }: { name: string; className?: string; compact?: boolean }) {
  return (
    <Avatar
      className={cn("size-9", compact ? "border-2" : "ring-2", className)}
      style={compact ? { borderColor: personColor(name) } : { ["--tw-ring-color" as string]: personColor(name) }}
      title={name}
    >
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
