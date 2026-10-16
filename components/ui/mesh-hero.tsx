"use client";
import dynamic from "next/dynamic";

import { cn } from "@/lib/utils";

// Loaded on demand and never on the server: the shader (WebGL) code is its own chunk that only the
// pages using a hero download, and the page is fully usable before it arrives.
const MeshBackdrop = dynamic(() => import("./mesh-backdrop"), { ssr: false });

/**
 * A card with a soft animated mesh-gradient behind its content. The CSS gradient underneath is what
 * you see first (and what stays if WebGL is unavailable or still loading), so it never looks unfinished.
 */
export function MeshHero({
  colors,
  className,
  children,
}: {
  colors: string[];
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <div
      className={cn("relative isolate overflow-hidden", className)}
      style={{ backgroundImage: `radial-gradient(120% 120% at 100% 0%, ${colors[2] ?? colors[0]} 0%, ${colors[1] ?? colors[0]} 45%, ${colors[0]} 100%)` }}
    >
      <MeshBackdrop colors={colors} />
      {/* a scrim keeps the text readable over the brightest part of the gradient */}
      <div aria-hidden className="absolute inset-0 bg-gradient-to-t from-black/45 via-black/10 to-transparent" />
      <div className="relative">{children}</div>
    </div>
  );
}
