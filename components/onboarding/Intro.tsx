"use client";
import { useEffect, useState } from "react";
import { AnimatePresence, m } from "motion/react";
import { Camera, Scale, Wallet } from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

const SLIDES = [
  {
    icon: Camera,
    title: "Scan it or type it",
    body: "Snap a receipt and we read the items, or just enter an amount for a quick split.",
  },
  {
    icon: Scale,
    title: "Split it fairly",
    body: "Equally, by sliders, by percent, or in plain words like “Sam didn’t have the rice”.",
  },
  {
    icon: Wallet,
    title: "Get paid back",
    body: "Track who has paid, nudge people on WhatsApp, and share a receipt with your DuitNow QR on it.",
  },
] as const;

const SWIPE = 60; // px of horizontal drag that counts as a swipe

// First-run welcome: three swipeable screens, skippable at any point. The caller remembers that it has been
// seen (see `useIntro`), so it only ever shows once per device.
export function Intro({ onDone }: { onDone: () => void }) {
  const [index, setIndex] = useState(0);
  const [dir, setDir] = useState(1);
  const last = index === SLIDES.length - 1;
  const slide = SLIDES[index];

  const go = (next: number) => {
    if (next < 0 || next >= SLIDES.length) return;
    setDir(next > index ? 1 : -1);
    setIndex(next);
  };
  const advance = () => (last ? onDone() : go(index + 1));

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onDone();
      if (e.key === "ArrowRight") go(index + 1);
      if (e.key === "ArrowLeft") go(index - 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  return (
    <div role="dialog" aria-modal="true" aria-label="Welcome to Bill.a" data-testid="intro" className="fixed inset-0 z-[70] flex flex-col bg-background/80 text-foreground backdrop-blur-2xl">
      <div className="flex justify-end p-4">
        <button type="button" data-testid="intro-skip" onClick={onDone} className="min-h-11 rounded-full px-4 text-sm font-semibold text-muted-foreground transition-colors hover:text-foreground">
          Skip
        </button>
      </div>

      <div className="relative mx-auto flex w-full max-w-md flex-1 items-center overflow-hidden px-8">
        <AnimatePresence mode="wait" custom={dir} initial={false}>
          <m.div
            key={index}
            data-testid={`intro-slide-${index}`}
            custom={dir}
            variants={{ enter: (d: number) => ({ x: 60 * d, opacity: 0 }), center: { x: 0, opacity: 1 }, exit: (d: number) => ({ x: -60 * d, opacity: 0 }) }}
            initial="enter"
            animate="center"
            exit="exit"
            transition={{ duration: 0.22, ease: "easeOut" }}
            drag="x"
            dragConstraints={{ left: 0, right: 0 }}
            dragElastic={0.25}
            onDragEnd={(_, info) => {
              if (info.offset.x < -SWIPE) advance();
              else if (info.offset.x > SWIPE) go(index - 1);
            }}
            className="w-full touch-pan-y space-y-8 text-center"
          >
            <span className="mx-auto flex size-28 items-center justify-center rounded-[2rem] bg-coral/15 text-coral shadow-[var(--coral-glow)]">
              <slide.icon className="size-12" strokeWidth={1.75} />
            </span>
            <div className="space-y-3">
              <h2 className="type-title">{slide.title}</h2>
              <p className="text-base leading-relaxed text-muted-foreground">{slide.body}</p>
            </div>
          </m.div>
        </AnimatePresence>
      </div>

      <div className="mx-auto w-full max-w-md space-y-5 px-8 pb-[max(2rem,env(safe-area-inset-bottom))]">
        <div className="flex justify-center gap-2" role="tablist" aria-label="Welcome screens">
          {SLIDES.map((s, i) => (
            <button
              key={s.title}
              type="button"
              role="tab"
              aria-selected={i === index}
              aria-label={`Screen ${i + 1} of ${SLIDES.length}`}
              onClick={() => go(i)}
              className="flex size-6 items-center justify-center"
            >
              <span className={cn("h-2 rounded-full transition-all", i === index ? "w-6 bg-coral" : "w-2 bg-white/20")} />
            </button>
          ))}
        </div>
        <Button type="button" size="lg" data-testid="intro-next" autoFocus onClick={advance} className="h-14 w-full text-base">
          {last ? "Get started" : "Next"}
        </Button>
      </div>
    </div>
  );
}

const KEY = "billa.intro.v1";

/** Whether to show the intro: until this device has seen it. Asked after mount, so it is false on the first (server) render. */
export function useIntro(): { show: boolean; finish: () => void } {
  const [show, setShow] = useState(false);
  useEffect(() => {
    // After mount, so the server-rendered page never disagrees with this device's storage.
    queueMicrotask(() => {
      try {
        if (localStorage.getItem(KEY) !== "1") setShow(true);
      } catch {
        /* storage unavailable (private window): skip the intro rather than show it every time */
      }
    });
  }, []);
  const finish = () => {
    setShow(false);
    try {
      localStorage.setItem(KEY, "1");
    } catch {
      /* not remembered */
    }
  };
  return { show, finish };
}
