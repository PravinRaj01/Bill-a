"use client";
import { useEffect } from "react";

import { savePushSubscription } from "@/lib/actions/push";

const KEY = "billa.push.resynced";
const DAY = 24 * 60 * 60 * 1000;

// Keeps this device's push subscription known to the server. Browsers rotate subscriptions and the server drops ones
// the push service reports gone, so a phone that says "on" could quietly stop receiving. Once a day, if this device
// has permission and a subscription, tell the server again (saving is an idempotent upsert). Renders nothing, never
// throws, never asks for permission.
export function PushResync({ signedIn }: { signedIn: boolean }) {
  useEffect(() => {
    if (!signedIn || !("serviceWorker" in navigator) || !("PushManager" in window) || !("Notification" in window)) return;
    if (Notification.permission !== "granted") return;
    try {
      if (Date.now() - Number(localStorage.getItem(KEY) ?? 0) < DAY) return;
    } catch {
      return;
    }
    let cancelled = false;
    (async () => {
      const reg = await navigator.serviceWorker.getRegistration();
      const sub = await reg?.pushManager.getSubscription();
      if (!sub || cancelled) return;
      await savePushSubscription(sub.toJSON());
      try {
        localStorage.setItem(KEY, String(Date.now()));
      } catch {
        /* resyncs next time */
      }
    })().catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [signedIn]);
  return null;
}
