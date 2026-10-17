"use client";
import { Suspense, useState } from "react";
import { AnimatePresence, m } from "motion/react";
import { ChevronLeft, Loader2, Sparkles } from "lucide-react";

import { Button } from "@/components/ui/button";
import { PageTransition } from "@/components/page-transition";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { ApiKeySettings } from "@/components/ai/ApiKeySettings";
import { AmountStep } from "@/components/split/steps/AmountStep";
import { HowStep } from "@/components/split/steps/HowStep";
import { NamesStep } from "@/components/split/steps/NamesStep";
import { QuickSplitStep } from "@/components/split/steps/QuickSplitStep";
import { ScanStep } from "@/components/split/steps/ScanStep";
import { ReviewStep } from "@/components/split/steps/ReviewStep";
import { SummaryStep } from "@/components/split/steps/SummaryStep";
import { useSplitSession, type Step } from "@/hooks/use-split-session";

// The new-session flow: NAMES -> SCAN -> REVIEW -> SUMMARY. All state and behaviour live in
// useSplitSession(); each step is a view over it. This file only wires the shell together.
function BillSplitterContent() {
  const s = useSplitSession();

  // Steps slide in the direction of travel (forward = from the right). Reduced motion is
  // handled globally by <MotionConfig reducedMotion="user">.
  const [nav, setNav] = useState<{ step: Step; dir: 1 | -1 }>({ step: s.step, dir: 1 });
  if (nav.step !== s.step) {
    setNav({ step: s.step, dir: s.stepOrder.indexOf(s.step) >= s.stepOrder.indexOf(nav.step) ? 1 : -1 });
  }
  const dir = nav.dir;

  const wide = s.step === "REVIEW" && s.showPhoto && !!s.photoUrl;

  return (
    <PageTransition>
    <main className={`mx-auto mb-20 flex w-full max-w-xl flex-1 flex-col gap-6 p-6 ${s.step === "REVIEW" ? "lg:max-w-5xl" : ""} ${wide ? "2xl:max-w-7xl" : ""}`}>
      {s.step !== s.firstStep && (
        <div className="flex items-center justify-between">
          <Button variant="ghost" className="h-11 w-fit px-1 text-sm font-semibold text-muted-foreground hover:bg-transparent hover:text-foreground" onClick={s.back}>
            <ChevronLeft size={16} className="mr-1" /> Back
          </Button>
          {/* AI keys only matter for scanning and describing a split; quick split is plain arithmetic. */}
          {!s.amountFlow && s.step !== "HOW" && <Button variant="ghost" data-testid="open-ai-settings" className="h-11 px-1 text-sm font-semibold text-muted-foreground hover:bg-transparent hover:text-foreground" onClick={() => s.setKeySheetOpen(true)}>
            <Sparkles size={14} className="mr-1.5 text-warning" /> AI settings{s.keys.groq || s.keys.gemini ? " ✓" : ""}
          </Button>}
        </div>
      )}

      <Sheet open={s.keySheetOpen} onOpenChange={s.setKeySheetOpen}>
        <SheetContent side="bottom" className="glass-strong max-h-[88vh] overflow-y-auto rounded-t-[28px] text-foreground sm:mx-auto sm:max-w-xl">
          <SheetHeader>
            <SheetTitle className="text-foreground">AI settings</SheetTitle>
            <SheetDescription className="text-muted-foreground">Bring your own free API key for smarter splits.</SheetDescription>
          </SheetHeader>
          <div className="px-4 pb-6">
            <ApiKeySettings />
          </div>
        </SheetContent>
      </Sheet>

      <AnimatePresence mode="wait" initial={false}>
        <m.div
          key={s.step}
          initial={{ opacity: 0, x: dir * 24 }}
          animate={{ opacity: 1, x: 0, transition: { duration: 0.22, ease: "easeOut" } }}
          exit={{ opacity: 0, x: dir * -24, transition: { duration: 0.12, ease: "easeIn" } }}
        >
          {s.step === "AMOUNT" && <AmountStep s={s} />}
          {s.step === "NAMES" && <NamesStep s={s} />}
          {s.step === "HOW" && <HowStep s={s} />}
          {s.step === "SCAN" && <ScanStep s={s} />}
          {s.step === "REVIEW" && <ReviewStep s={s} />}
          {s.step === "SPLIT" && <QuickSplitStep s={s} />}
          {s.step === "SUMMARY" && <SummaryStep s={s} />}
        </m.div>
      </AnimatePresence>
    </main>
    </PageTransition>
  );
}

export default function BillSplitter() {
  return (
    <Suspense fallback={<div className="flex min-h-screen items-center justify-center bg-background"><Loader2 className="size-8 animate-spin text-foreground opacity-20" /></div>}>
      <BillSplitterContent />
    </Suspense>
  );
}
