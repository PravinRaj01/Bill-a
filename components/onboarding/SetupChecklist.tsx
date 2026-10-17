"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { Check, ChevronRight, Loader2, X } from "lucide-react";
import { toast } from "sonner";

import { Input } from "@/components/ui/input";
import { updateProfileName } from "@/lib/actions/user";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

const KEY = "billa.setup.dismissed.v1";

// "Get set up" on Home: three things that make the app yours (your name, how friends pay you, a saved
// group). Each ticks itself from real data, can be dismissed for good, and the card disappears once all
// three are done.
export function SetupChecklist({
  hasName,
  hasPayment,
  hasGroup,
  onNameSaved,
}: {
  hasName: boolean;
  hasPayment: boolean;
  hasGroup: boolean;
  onNameSaved: (name: string | null) => void;
}) {
  const [dismissed, setDismissed] = useState(true); // hidden until the device's choice is known
  const [name, setName] = useState("");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    queueMicrotask(() => {
      try {
        setDismissed(localStorage.getItem(KEY) === "1");
      } catch {
        setDismissed(false);
      }
    });
  }, []);

  const done = [hasName, hasPayment, hasGroup].filter(Boolean).length;
  if (dismissed || done === 3) return null;

  const dismiss = () => {
    setDismissed(true);
    try {
      localStorage.setItem(KEY, "1");
    } catch {
      /* not remembered */
    }
  };

  const saveName = async () => {
    const n = name.trim();
    if (!n) return;
    setSaving(true);
    try {
      const r = await updateProfileName(n);
      if (!r.ok) toast.error(r.error);
      else onNameSaved(r.name);
    } catch {
      toast.error("Couldn't save your name. Please try again.");
    } finally {
      setSaving(false);
    }
  };

  const row = "flex min-h-14 items-center gap-3 rounded-2xl px-3 py-2";
  const tick = (ok: boolean) => (
    <span className={cn("flex size-7 shrink-0 items-center justify-center rounded-full border", ok ? "border-success bg-success/15 text-success" : "border-white/15 text-transparent")}>
      <Check size={14} strokeWidth={3} />
    </span>
  );

  return (
    <section aria-label="Get set up" data-testid="setup-checklist" className="space-y-2 rounded-3xl glass p-4">
      <div className="flex items-center justify-between px-3">
        <div>
          <h2 className="text-base font-bold">Get set up</h2>
          <p className="text-xs text-muted-foreground" data-testid="setup-progress">{done} of 3 done</p>
        </div>
        <button type="button" data-testid="setup-dismiss" aria-label="Dismiss" onClick={dismiss} className="flex size-11 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-white/5 hover:text-foreground">
          <X size={18} />
        </button>
      </div>

      <ul>
        <li data-testid="setup-name" data-done={hasName}>
          <div className={row}>
            {tick(hasName)}
            {hasName ? (
              <span className="text-sm text-muted-foreground line-through">Say who you are</span>
            ) : (
              <div className="flex min-w-0 flex-1 gap-2">
                <Input
                  data-testid="setup-name-input"
                  aria-label="Your name"
                  placeholder="Your name"
                  maxLength={80}
                  autoComplete="name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && saveName()}
                  enterKeyHint="done"
                  className="h-11 rounded-xl border-white/10 bg-white/10 text-base text-foreground"
                />
                <Button type="button" data-testid="setup-name-save" onClick={saveName} disabled={!name.trim() || saving}>
                  {saving ? <Loader2 className="size-4 animate-spin" /> : "Save"}
                </Button>
              </div>
            )}
          </div>
        </li>

        <li data-testid="setup-payment" data-done={hasPayment}>
          {hasPayment ? (
            <div className={row}>
              {tick(true)}
              <span className="text-sm text-muted-foreground line-through">Add how friends pay you</span>
            </div>
          ) : (
            <Link href="/dashboard/account?tab=pay" transitionTypes={["nav-forward"]} className={cn(row, "transition-colors hover:bg-white/5")}>
              {tick(false)}
              <span className="flex-1 text-sm font-semibold">Add how friends pay you</span>
              <ChevronRight size={16} className="text-muted-foreground" />
            </Link>
          )}
        </li>

        <li data-testid="setup-group" data-done={hasGroup}>
          {hasGroup ? (
            <div className={row}>
              {tick(true)}
              <span className="text-sm text-muted-foreground line-through">Save a group</span>
            </div>
          ) : (
            <Link href="/dashboard/new" transitionTypes={["nav-forward"]} className={cn(row, "transition-colors hover:bg-white/5")}>
              {tick(false)}
              <span className="flex-1">
                <span className="block text-sm font-semibold">Save a group</span>
                <span className="block text-xs text-muted-foreground">On your next split, choose &ldquo;Save this group&rdquo;</span>
              </span>
              <ChevronRight size={16} className="text-muted-foreground" />
            </Link>
          )}
        </li>
      </ul>
    </section>
  );
}
