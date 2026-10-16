"use client";
import { useEffect, useRef, useState } from "react";
import { MeshGradient } from "@paper-design/shaders-react";

// The WebGL half of <MeshHero>. Only ever loaded on the client (next/dynamic, ssr: false), so it
// can read the device here. It animates only when that is welcome, and is otherwise a still frame:
//   - the user prefers reduced motion, has Data Saver on, or the device is low-power (<= 4 cores)
//   - the tile is off screen, or the tab is hidden (no GPU work nobody can see)

interface NetworkInformation {
  saveData?: boolean;
}

function shouldAnimate(): boolean {
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const saveData = (navigator as Navigator & { connection?: NetworkInformation }).connection?.saveData === true;
  const weak = (navigator.hardwareConcurrency ?? 8) <= 4;
  return !reduced && !saveData && !weak;
}

export default function MeshBackdrop({ colors }: { colors: string[] }) {
  const ref = useRef<HTMLDivElement>(null);
  const [animate] = useState(shouldAnimate);
  const [onScreen, setOnScreen] = useState(true);
  const [tabVisible, setTabVisible] = useState(true);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(([entry]) => setOnScreen(entry.isIntersecting));
    io.observe(el);
    const onVisibility = () => setTabVisible(!document.hidden);
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      io.disconnect();
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, []);

  return (
    <div ref={ref} aria-hidden className="absolute inset-0">
      <MeshGradient
        className="size-full"
        colors={colors}
        distortion={0.8}
        swirl={0.15}
        grainMixer={0}
        grainOverlay={0}
        speed={animate && onScreen && tabVisible ? 0.15 : 0}
        maxPixelCount={500_000}
      />
    </div>
  );
}
