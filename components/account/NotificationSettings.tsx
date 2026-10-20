"use client";

import { useEffect, useState } from "react";
import { BellRing, Share } from "lucide-react";
import { toast } from "sonner";

import { Switch } from "@/components/ui/switch";
import { getPushState, savePushSubscription, sendTestPush } from "@/lib/actions/push";
import { dropThisDevicePush } from "@/lib/push/client";

type Status = "loading" | "unavailable" | "unsupported" | "needs-install" | "blocked" | "off" | "on";

const isIos = () => /iphone|ipad|ipod/i.test(navigator.userAgent);
const isInstalled = () => window.matchMedia("(display-mode: standalone)").matches || (navigator as { standalone?: boolean }).standalone === true;

function keyBytes(b64: string): Uint8Array<ArrayBuffer> {
  const raw = atob(b64.replace(/-/g, "+").replace(/_/g, "/").padEnd(Math.ceil(b64.length / 4) * 4, "="));
  return Uint8Array.from(raw, (c) => c.charCodeAt(0));
}

/** The registration the page's service worker made (null in dev, where the worker is off). */
const registration = () => navigator.serviceWorker?.getRegistration() ?? Promise.resolve(undefined);

// Push is opt-in: nothing is asked until the switch is tapped. On iPhone it only exists once the app is on the
// Home Screen, so that case explains how instead of showing a switch that can't work.
export function NotificationSettings() {
  const [status, setStatus] = useState<Status>("loading");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const set = (s: Status) => !cancelled && setStatus(s);
      const state = await getPushState().catch(() => null);
      if (!state?.available) return set("unavailable");
      if (isIos() && !isInstalled()) return set("needs-install");
      if (!("serviceWorker" in navigator) || !("PushManager" in window) || !("Notification" in window)) return set("unsupported");
      if (Notification.permission === "denied") return set("blocked");
      const reg = await registration();
      if (!reg) return set("unsupported");
      set((await reg.pushManager.getSubscription()) ? "on" : "off");
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  async function turnOn() {
    const state = await getPushState();
    const reg = await registration();
    if (!state.publicKey || !reg) throw new Error("unavailable");
    const permission = await Notification.requestPermission();
    // Closing the prompt ("default") is not a refusal: the switch stays usable. Only a real "denied" is blocked.
    if (permission === "denied") return setStatus("blocked");
    if (permission !== "granted") {
      toast("Notifications weren't turned on. You can try again any time.");
      return;
    }
    const sub = await reg.pushManager.subscribe({ userVisibleOnly: true, applicationServerKey: keyBytes(state.publicKey) });
    await savePushSubscription(sub.toJSON());
    setStatus("on");
    toast.success("Notifications on for this device");
  }

  async function turnOff() {
    await dropThisDevicePush();
    setStatus("off");
  }

  async function toggle(next: boolean) {
    setBusy(true);
    try {
      await (next ? turnOn() : turnOff());
    } catch {
      toast.error("Couldn't change notifications. Try again.");
    } finally {
      setBusy(false);
    }
  }

  async function test() {
    setBusy(true);
    try {
      const { delivered, ms } = await sendTestPush();
      if (delivered > 0) toast.success(`Test sent in ${ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`}`, { description: "If it shows up late, that delay is your phone's battery settings, not Bill.a." });
      else toast.error("No device got it. Switch notifications off and on again.");
    } catch {
      toast.error("Couldn't send a test. Try again.");
    } finally {
      setBusy(false);
    }
  }

  const note: Partial<Record<Status, string>> = {
    unavailable: "Notifications aren't set up on this server yet.",
    unsupported: "This browser can't show notifications. Try the installed app.",
    blocked: "Notifications are blocked for Bill.a. Allow them in your browser or phone settings, then come back.",
  };

  return (
    <section data-testid="account-notifications" className="space-y-3">
      <div className="flex items-center justify-between gap-4 rounded-2xl glass p-5">
        <div className="min-w-0 space-y-1">
          <h2 className="flex items-center gap-2 text-sm font-bold text-foreground"><BellRing size={16} /> Notifications on this device</h2>
          <p className="text-xs text-muted-foreground">Friend requests, shared bills and payments, plus a short summary each morning when money is still owed.</p>
        </div>
        <Switch
          aria-label="Notifications on this device"
          checked={status === "on"}
          disabled={busy || !["off", "on"].includes(status)}
          onCheckedChange={toggle}
        />
      </div>

      {status === "on" && (
        <button type="button" onClick={test} disabled={busy} className="px-1 text-xs text-muted-foreground underline underline-offset-4 hover:text-foreground disabled:opacity-50">
          Send me a test notification
        </button>
      )}

      {status === "needs-install" && (
        <div className="space-y-2 rounded-2xl glass p-5 text-sm text-muted-foreground">
          <p className="font-semibold text-foreground">On iPhone, add Bill.a to your Home Screen first</p>
          <ol className="list-decimal space-y-1 pl-5">
            <li>Open this page in Safari and tap <Share size={14} className="inline align-text-bottom" /> Share.</li>
            <li>Choose <strong>Add to Home Screen</strong>, then open Bill.a from there.</li>
            <li>Come back to Account and switch notifications on.</li>
          </ol>
        </div>
      )}
      {note[status] && <p className="px-1 text-xs text-muted-foreground">{note[status]}</p>}
    </section>
  );
}
