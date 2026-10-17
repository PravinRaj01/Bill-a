import * as React from "react"

import { cn } from "@/lib/utils"

// A recessed glass well, 48px tall, 16px text (so iOS doesn't zoom on focus).
function Input({ className, type, ...props }: React.ComponentProps<"input">) {
  return (
    <input
      type={type}
      data-slot="input"
      className={cn(
        "glass-inset h-12 w-full min-w-0 rounded-xl px-4 py-2 text-base text-foreground transition-[color,box-shadow,border-color] outline-none",
        "placeholder:text-muted-foreground selection:bg-primary selection:text-primary-foreground",
        "file:inline-flex file:h-7 file:border-0 file:bg-transparent file:text-sm file:font-medium file:text-foreground",
        "focus-visible:border-primary/60 focus-visible:ring-[3px] focus-visible:ring-ring",
        "disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-50",
        "aria-invalid:border-destructive aria-invalid:ring-destructive/25",
        className
      )}
      {...props}
    />
  )
}

export { Input }
