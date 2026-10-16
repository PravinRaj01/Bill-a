"use client";
import { LazyMotion, MotionConfig, domAnimation } from "motion/react";

// One place to configure Motion for the whole app:
//  - LazyMotion + domAnimation keeps the initial bundle small (use `m.*`, not `motion.*`).
//  - reducedMotion="user" makes every transform/layout animation respect the OS setting.
export function MotionProvider({ children }: { children: React.ReactNode }) {
  return (
    <LazyMotion features={domAnimation} strict>
      <MotionConfig reducedMotion="user">{children}</MotionConfig>
    </LazyMotion>
  );
}
