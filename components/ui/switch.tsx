"use client"

import * as React from "react"
import * as SwitchPrimitive from "@radix-ui/react-switch"

import { cn } from "@/lib/utils"

// Coral when on, 48x28 visible with a 44px+ hit area (the ::after), so it works with a thumb.
function Switch({
  className,
  ...props
}: React.ComponentProps<typeof SwitchPrimitive.Root>) {
  return (
    <SwitchPrimitive.Root
      data-slot="switch"
      className={cn(
        "peer relative inline-flex h-7 w-12 shrink-0 items-center rounded-full border border-white/10 transition-colors outline-none",
        "data-[state=checked]:bg-primary data-[state=checked]:shadow-[var(--coral-glow)] data-[state=unchecked]:bg-white/15",
        "focus-visible:ring-[3px] focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50",
        "after:absolute after:-inset-2.5 after:content-['']",
        className
      )}
      {...props}
    >
      <SwitchPrimitive.Thumb
        data-slot="switch-thumb"
        className="pointer-events-none block size-5 rounded-full bg-white shadow-md ring-0 transition-transform data-[state=checked]:translate-x-6 data-[state=unchecked]:translate-x-1"
      />
    </SwitchPrimitive.Root>
  )
}

export { Switch }
