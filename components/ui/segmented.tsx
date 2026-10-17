"use client";
import * as React from "react";
import { m } from "motion/react";

import { cn } from "@/lib/utils";

export interface SegmentedOption<T extends string> {
  value: T;
  label: string;
  icon?: React.ReactNode;
}

// A radiogroup with a sliding pill (Motion layoutId). Arrow keys move between options.
export function Segmented<T extends string>({
  value,
  onValueChange,
  options,
  label,
  className,
}: {
  value: T;
  onValueChange: (v: T) => void;
  options: SegmentedOption<T>[];
  /** Accessible name for the group. */
  label: string;
  className?: string;
}) {
  const id = React.useId();
  const onKeyDown = (e: React.KeyboardEvent) => {
    const i = options.findIndex((o) => o.value === value);
    const step = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 0;
    if (!step) return;
    e.preventDefault();
    onValueChange(options[(i + step + options.length) % options.length].value);
  };

  return (
    <div
      role="radiogroup"
      aria-label={label}
      onKeyDown={onKeyDown}
      className={cn("relative flex w-full glass-inset rounded-full p-1", className)}
    >
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={active}
            tabIndex={active ? 0 : -1}
            onClick={() => onValueChange(o.value)}
            className={cn(
              "relative z-10 flex min-h-11 flex-1 items-center justify-center gap-1.5 rounded-full px-3 text-sm font-semibold whitespace-nowrap transition-colors outline-none focus-visible:ring-2 focus-visible:ring-white/30",
              active ? "text-coral-foreground" : "text-muted-foreground hover:text-foreground",
            )}
          >
            {active && (
              <m.span
                layoutId={`seg-${id}`}
                className="absolute inset-0 -z-10 rounded-full bg-coral shadow-[var(--coral-glow)]"
                transition={{ type: "spring", stiffness: 500, damping: 40 }}
              />
            )}
            {o.icon}
            {o.label}
          </button>
        );
      })}
    </div>
  );
}
