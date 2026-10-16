"use client";
import { useState } from "react";
import { toast } from "sonner";

import { Slider } from "@/components/ui/slider";
import { Segmented } from "@/components/ui/segmented";
import { Money } from "@/components/ui/money";
import { AvatarStack, PersonAvatar } from "@/components/person-avatar";
import { haptic } from "@/lib/ui/haptics";
import { personColor } from "@/lib/ui/person-color";

// Visual/behaviour sandbox for the shared UI building blocks (dev only: see app/dev/layout.tsx).
// Used for manual checks and Playwright smoke tests at phone and tablet widths.
const PEOPLE = ["Pravin", "Lakx", "Wei Jie", "Aisha"];

export default function UiKit() {
  const [mode, setMode] = useState<"equal" | "custom" | "describe">("equal");
  const [pct, setPct] = useState(70);
  const total = 7770;
  const mine = Math.round((total * pct) / 100);

  return (
    <main className="mx-auto flex w-full max-w-xl flex-col gap-8 p-6 pb-32">
      <h1 className="text-2xl font-bold">UI kit</h1>

      <section className="space-y-3">
        <h2 className="text-sm text-muted-foreground">Segmented</h2>
        <Segmented
          label="How to split"
          value={mode}
          onValueChange={setMode}
          options={[
            { value: "equal", label: "Equal" },
            { value: "custom", label: "Custom" },
            { value: "describe", label: "Describe" },
          ]}
        />
        <p data-testid="mode">{mode}</p>
      </section>

      <section className="space-y-3 rounded-3xl bg-surface p-4">
        <h2 className="text-sm text-muted-foreground">Slider + Money</h2>
        <div className="flex items-center justify-between gap-3">
          <PersonAvatar name="Pravin" />
          <span className="flex-1 font-semibold">Pravin</span>
          <span data-testid="pct" className="tabular-nums">{pct}%</span>
          <Money cents={mine} className="text-lg font-bold" />
        </div>
        <Slider
          aria-label="Pravin's share"
          value={[pct]}
          max={100}
          step={1}
          onValueChange={([v]) => {
            if (v % 5 === 0 && v !== pct) haptic("selection");
            setPct(v);
          }}
          style={{ ["--slider-color" as string]: personColor("Pravin") }}
        />
        <div className="flex items-center justify-between text-sm text-muted-foreground">
          <span>Lakx {100 - pct}%</span>
          <Money cents={total - mine} />
        </div>
      </section>

      <section className="space-y-3">
        <h2 className="text-sm text-muted-foreground">Avatars</h2>
        <div className="flex gap-2">
          {PEOPLE.map((p) => (
            <PersonAvatar key={p} name={p} />
          ))}
        </div>
        <AvatarStack names={[...PEOPLE, "Mei", "Sam"]} />
      </section>

      <section className="space-y-3">
        <h2 className="text-sm text-muted-foreground">Toast</h2>
        <button
          type="button"
          onClick={() => toast("Shares reset to equal", { action: { label: "Undo", onClick: () => {} } })}
          className="min-h-11 rounded-full bg-coral px-5 font-bold text-coral-foreground shadow-[var(--coral-glow)]"
        >
          Show toast
        </button>
      </section>
    </main>
  );
}
