"use client";
import { useEffect, useState } from "react";
import { toast } from "sonner";

import { Switch } from "@/components/ui/switch";
import { getNotifyPrefs, setNotifyPref } from "@/lib/actions/push";
import { CATEGORIES, CATEGORY_INFO, isCategoryOn, type Category, type NotifyPrefs } from "@/lib/notifications/prefs";

// What reaches the phone, one switch per kind of notification. Switching one off only stops the phone alert: the
// inbox keeps its record. (The daily summary is the exception: off means it isn't made at all.)
export function NotificationPrefs() {
  const [prefs, setPrefs] = useState<NotifyPrefs | null>(null);
  const [busy, setBusy] = useState<Category | null>(null);

  useEffect(() => {
    getNotifyPrefs().then(setPrefs).catch(() => setPrefs({}));
  }, []);

  const flip = async (category: Category, on: boolean) => {
    const before = prefs;
    setPrefs((p) => ({ ...(p ?? {}), [category]: on }));
    setBusy(category);
    try {
      setPrefs(await setNotifyPref(category, on));
    } catch {
      setPrefs(before);
      toast.error("Couldn't save that. Try again.");
    } finally {
      setBusy(null);
    }
  };

  return (
    <section data-testid="notify-prefs" aria-label="What reaches your phone" className="space-y-3">
      <div className="space-y-1 px-1">
        <h2 className="type-eyebrow">What reaches your phone</h2>
        <p className="text-xs text-muted-foreground">Switching one off only stops the phone alert. It still appears in your inbox.</p>
      </div>
      <ul className="space-y-2">
        {CATEGORIES.map((c) => (
          <li key={c} className="flex items-center justify-between gap-4 rounded-2xl glass p-4">
            <span className="min-w-0 space-y-0.5">
              <span className="block text-sm font-bold text-foreground">{CATEGORY_INFO[c].title}</span>
              <span className="block text-xs text-muted-foreground">{CATEGORY_INFO[c].blurb}</span>
            </span>
            <Switch
              data-testid={`pref-${c}`}
              aria-label={CATEGORY_INFO[c].title}
              checked={prefs ? isCategoryOn(prefs, c) : true}
              disabled={prefs === null || busy === c}
              onCheckedChange={(on) => void flip(c, on)}
            />
          </li>
        ))}
      </ul>
    </section>
  );
}
