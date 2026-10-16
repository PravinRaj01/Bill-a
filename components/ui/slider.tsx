"use client";
import * as React from "react";
import * as SliderPrimitive from "@radix-ui/react-slider";

import { cn } from "@/lib/utils";

// Touch-first: the visible thumb is 28px but its hit area is 44px (the after: pseudo-element),
// and the track row is 44px tall. `--slider-color` sets the fill and the thumb ring, so a
// person's slider can wear their colour: style={{ "--slider-color": personColor(name) }}.
//
// Radix puts role="slider" on the THUMB, so the accessible name and the spoken value have to
// go there: `aria-label` names it ("Pravin's share") and `valueText` is what a screen reader
// says ("Pravin, 70 percent, RM77.70") instead of a bare number of cents.
function Slider({
  className,
  valueText,
  ...props
}: React.ComponentProps<typeof SliderPrimitive.Root> & { valueText?: string }) {
  const thumbs = props.value ?? props.defaultValue ?? [0];
  const label = props["aria-label"];
  return (
    <SliderPrimitive.Root
      data-slot="slider"
      className={cn(
        "relative flex h-11 w-full touch-none items-center select-none data-[disabled]:opacity-50",
        className,
      )}
      {...props}
    >
      <SliderPrimitive.Track
        data-slot="slider-track"
        className="relative h-2.5 w-full grow overflow-hidden rounded-full bg-surface-2"
      >
        <SliderPrimitive.Range
          data-slot="slider-range"
          className="absolute h-full rounded-full bg-[var(--slider-color,var(--coral))]"
        />
      </SliderPrimitive.Track>
      {thumbs.map((_, i) => (
        <SliderPrimitive.Thumb
          key={i}
          data-slot="slider-thumb"
          aria-label={label}
          aria-valuetext={valueText}
          className="block size-7 rounded-full border-2 border-[var(--slider-color,var(--coral))] bg-white shadow-lg outline-none transition-[box-shadow,transform] focus-visible:ring-4 focus-visible:ring-white/20 active:scale-110 after:absolute after:-inset-2 after:content-['']"
        />
      ))}
    </SliderPrimitive.Root>
  );
}

export { Slider };
