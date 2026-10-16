"use client";
import { LazyMotion, MotionConfig } from "motion/react";

// One place to configure Motion for the whole app:
//  - LazyMotion keeps the initial bundle small (use `m.*`, not `motion.*`). The features are loaded after
//    first paint; domMax is needed (not just domAnimation) for drag and for layout / layoutId animations.
//  - reducedMotion="user" makes every transform/layout animation respect the OS setting.
const loadFeatures = () => import("./motion-features").then((mod) => mod.default);

export function MotionProvider({ children }: { children: React.ReactNode }) {
  return (
    <LazyMotion features={loadFeatures} strict>
      <MotionConfig reducedMotion="user">{children}</MotionConfig>
    </LazyMotion>
  );
}
