"use client";
import { useState } from "react";
import { Plus, Share2, Trash2 } from "lucide-react";
import { toast } from "sonner";

import { AvatarStack, PersonAvatar } from "@/components/person-avatar";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Drawer, DrawerContent, DrawerDescription, DrawerHeader, DrawerTitle } from "@/components/ui/drawer";
import { Input } from "@/components/ui/input";
import { Money } from "@/components/ui/money";
import { Segmented } from "@/components/ui/segmented";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { haptic } from "@/lib/ui/haptics";

// The living reference for the Graphite Glass + Coral theme (dev only: see app/dev/layout.tsx). Every shared
// primitive appears here, so a change to a token or a primitive can be checked in one place, at phone and
// tablet widths, and used by the Playwright smoke tests.
const PEOPLE = ["Alexis", "Riley", "Wei Jie", "Aisha"];
const VARIANTS = ["default", "glass", "outline", "secondary", "ghost", "destructive", "whatsapp"] as const;

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="space-y-3">
      <h2 className="type-eyebrow">{title}</h2>
      {children}
    </section>
  );
}

export default function UiKit() {
  const [mode, setMode] = useState<"equal" | "custom" | "describe">("equal");
  const [pct, setPct] = useState(70);
  const [on, setOn] = useState(true);
  const [sheet, setSheet] = useState(false);
  const total = 7770;
  const mine = Math.round((total * pct) / 100);

  return (
    <main className="mx-auto flex w-full max-w-xl flex-col gap-8 p-6 pb-32 text-foreground">
      <header className="space-y-1">
        <p className="type-eyebrow">Graphite Glass + Coral</p>
        <h1 data-testid="kit-title" className="type-title">UI kit</h1>
      </header>

      <Section title="Glass">
        <div className="grid gap-3">
          <div data-testid="kit-glass" className="glass rounded-3xl p-5">
            <p className="type-card-title">glass</p>
            <p className="text-sm text-muted-foreground">Cards, rows and tiles. It blurs the orbs behind it.</p>
          </div>
          <div data-testid="kit-glass-strong" className="glass-strong rounded-3xl p-5">
            <p className="type-card-title">glass-strong</p>
            <p className="text-sm text-muted-foreground">Sheets, the nav, the header, toasts.</p>
          </div>
          <div className="glass rounded-3xl p-4">
            <div className="glass-inset rounded-2xl p-4 text-sm text-muted-foreground">glass-inset: a recessed well inside a panel.</div>
          </div>
        </div>
      </Section>

      <Section title="Buttons">
        <div className="flex flex-wrap gap-3">
          {VARIANTS.map((v) => (
            <Button key={v} variant={v} data-testid={`kit-btn-${v}`}>
              {v}
            </Button>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <Button size="sm">Small</Button>
          <Button>Default</Button>
          <Button size="lg">Large</Button>
          <Button size="icon" aria-label="Add"><Plus /></Button>
          <Button size="icon" variant="glass" aria-label="Share"><Share2 /></Button>
          <Button size="icon" variant="ghost" aria-label="Delete"><Trash2 /></Button>
          <Button disabled>Disabled</Button>
        </div>
      </Section>

      <Section title="Inputs and switch">
        <Input data-testid="kit-input" placeholder="Person's name" />
        <Input defaultValue="Invalid" aria-invalid />
        <label className="glass flex min-h-14 items-center justify-between gap-3 rounded-2xl px-4">
          <span className="text-sm">Add my payment details</span>
          <Switch data-testid="kit-switch" checked={on} onCheckedChange={setOn} aria-label="Add my payment details" />
        </label>
      </Section>

      <Section title="Segmented">
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
        <p data-testid="mode" className="text-sm text-muted-foreground">{mode}</p>
      </Section>

      <Section title="Slider + Money">
        <Card className="space-y-3 p-4">
          <div className="flex items-center justify-between gap-3">
            <PersonAvatar name="Alexis" />
            <span className="flex-1 font-semibold">Alexis</span>
            <span data-testid="pct" className="tabular-nums">{pct}%</span>
            <Money cents={mine} className="text-lg font-bold" />
          </div>
          <Slider
            aria-label="Alexis's share"
            value={[pct]}
            max={100}
            step={1}
            onValueChange={([v]) => {
              if (v % 5 === 0 && v !== pct) haptic("selection");
              setPct(v);
            }}
          />
          <div className="flex items-center justify-between text-sm text-muted-foreground">
            <span>Riley {100 - pct}%</span>
            <Money cents={total - mine} />
          </div>
        </Card>
      </Section>

      <Section title="Typography">
        <div className="space-y-2">
          <p className="type-eyebrow">eyebrow</p>
          <p className="type-title">Page title</p>
          <p className="type-card-title">Card title</p>
          <p className="text-sm text-muted-foreground">Body text on glass stays readable (muted foreground).</p>
        </div>
      </Section>

      <Section title="Status colours">
        <div className="flex flex-wrap gap-2 text-sm font-semibold">
          <span className="rounded-full bg-success/15 px-3 py-1 text-success">Settled ✓</span>
          <span className="rounded-full bg-warning/15 px-3 py-1 text-warning">Due today</span>
          <span className="rounded-full bg-destructive/15 px-3 py-1 text-destructive">Overdue</span>
          <span className="rounded-full bg-primary/15 px-3 py-1 text-primary">Coral</span>
        </div>
      </Section>

      <Section title="Avatars">
        <div className="flex gap-2">
          {PEOPLE.map((p) => (
            <PersonAvatar key={p} name={p} />
          ))}
        </div>
        <AvatarStack names={[...PEOPLE, "Mei", "Sam"]} />
      </Section>

      <Section title="Toast and sheet">
        <div className="flex gap-3">
          <Button data-testid="kit-toast" onClick={() => toast("Shares reset to equal", { action: { label: "Undo", onClick: () => {} } })}>
            Show toast
          </Button>
          <Button variant="glass" data-testid="kit-sheet-open" onClick={() => setSheet(true)}>
            Open sheet
          </Button>
        </div>
        <Drawer open={sheet} onOpenChange={setSheet}>
          <DrawerContent data-testid="kit-sheet">
            <DrawerHeader>
              <DrawerTitle>Frosted sheet</DrawerTitle>
              <DrawerDescription className="text-muted-foreground">Glass-strong over a blurred overlay.</DrawerDescription>
            </DrawerHeader>
            <div className="mx-auto w-full max-w-md space-y-3 px-4 pb-8">
              <Input placeholder="Group name" />
              <Button className="w-full" size="lg" onClick={() => setSheet(false)}>Save</Button>
            </div>
          </DrawerContent>
        </Drawer>
      </Section>
    </main>
  );
}
