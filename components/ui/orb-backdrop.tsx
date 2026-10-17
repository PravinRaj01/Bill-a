"use client";
import { useEffect, useState, type CSSProperties } from "react";

// The soft light behind every screen: five big radial-gradient orbs (coral, rose, graphite-blue) that drift
// very slowly. They sit fixed behind the app (see .orb-backdrop in globals.css), so frosted glass panels
// have something to blur. Only `transform` animates, the motion stops under prefers-reduced-motion (CSS),
// and it pauses while the tab is hidden.
const ORBS: { color: string; opacity: number; size: string; pos: CSSProperties; dx: string; dy: string; time: string }[] = [
  { color: "#ff6b57", opacity: 0.38, size: "clamp(420px, 62vmax, 880px)", pos: { top: "-18%", right: "-22%" }, dx: "-8vw", dy: "10vh", time: "90s" },
  { color: "#b8325a", opacity: 0.3, size: "clamp(380px, 55vmax, 800px)", pos: { top: "28%", left: "-26%" }, dx: "10vw", dy: "-6vh", time: "110s" },
  { color: "#ff8a5c", opacity: 0.2, size: "clamp(320px, 45vmax, 680px)", pos: { bottom: "-14%", right: "-12%" }, dx: "-10vw", dy: "-8vh", time: "100s" },
  { color: "#3c4f7c", opacity: 0.26, size: "clamp(380px, 55vmax, 780px)", pos: { bottom: "-20%", left: "-18%" }, dx: "8vw", dy: "-10vh", time: "120s" },
  { color: "#ff6b57", opacity: 0.16, size: "clamp(220px, 28vmax, 420px)", pos: { top: "46%", left: "42%" }, dx: "-6vw", dy: "6vh", time: "75s" },
];

export function OrbBackdrop() {
  const [paused, setPaused] = useState(false);
  useEffect(() => {
    const onVisibility = () => setPaused(document.hidden);
    document.addEventListener("visibilitychange", onVisibility);
    return () => document.removeEventListener("visibilitychange", onVisibility);
  }, []);

  return (
    <div aria-hidden data-testid="orb-backdrop" data-paused={paused} className="orb-backdrop">
      {ORBS.map((o, i) => (
        <span
          key={i}
          className="orb"
          style={
            {
              ...o.pos,
              width: o.size,
              height: o.size,
              "--orb-color": o.color,
              "--orb-opacity": o.opacity,
              "--orb-dx": o.dx,
              "--orb-dy": o.dy,
              "--orb-time": o.time,
            } as CSSProperties
          }
        />
      ))}
    </div>
  );
}
