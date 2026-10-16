"use client";
import { ViewTransition } from "react";

// Slides a page in and out when the navigation says which way it is going (see the nav-forward /
// nav-back rules in app/globals.css and the transitionTypes on the links). Anything untagged, like
// the side-nav tabs or the browser's Back button, is a plain swap: `default: "none"`.
//
// Wrap a page's own content with this, never a layout: layouts persist across navigations, so their
// enter/exit never fire. Browsers without View Transitions simply skip the animation.
const bySlide = { "nav-forward": "nav-forward", "nav-back": "nav-back", default: "none" } as const;

export function PageTransition({ children }: { children: React.ReactNode }) {
  return (
    <ViewTransition enter={bySlide} exit={bySlide} default="none">
      {children}
    </ViewTransition>
  );
}
