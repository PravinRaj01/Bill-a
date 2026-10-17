import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

// One button for the whole app. The default is the coral pill (the primary action); everything else is
// glass or ghost. Sizes are touch-first: nothing the user taps is under 44px except `sm`/`icon-sm`.
const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-full text-sm font-semibold transition-all disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-4 shrink-0 [&_svg]:shrink-0 outline-none focus-visible:ring-[3px] focus-visible:ring-ring active:scale-[0.98] aria-invalid:ring-destructive/30 aria-invalid:border-destructive",
  {
    variants: {
      variant: {
        // The primary action: solid coral with a glow.
        default: "bg-primary text-primary-foreground shadow-[var(--coral-glow)] hover:bg-primary/90",
        // Frosted secondary action.
        glass: "glass text-foreground hover:bg-white/10",
        // A hairline outline on a faint fill (kept for existing call sites).
        outline: "border border-white/12 bg-white/5 text-foreground hover:bg-white/10",
        secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80",
        ghost: "text-muted-foreground hover:bg-white/8 hover:text-foreground",
        destructive: "bg-destructive text-[#1a0b08] hover:bg-destructive/90 focus-visible:ring-destructive/30",
        // WhatsApp's own green: a brand exception, used only on "share" actions.
        whatsapp: "bg-[#25D366] text-black hover:bg-[#25D366]/90",
        link: "rounded-none text-primary underline-offset-4 hover:underline",
      },
      size: {
        default: "h-11 px-5 has-[>svg]:px-4",
        sm: "h-9 gap-1.5 px-4 text-[13px] has-[>svg]:px-3",
        lg: "h-12 px-6 text-base has-[>svg]:px-5",
        icon: "size-11",
        "icon-sm": "size-9",
        "icon-lg": "size-12",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

function Button({
  className,
  variant = "default",
  size = "default",
  asChild = false,
  ...props
}: React.ComponentProps<"button"> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean
  }) {
  const Comp = asChild ? Slot : "button"

  return (
    <Comp
      data-slot="button"
      data-variant={variant}
      data-size={size}
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  )
}

export { Button, buttonVariants }
