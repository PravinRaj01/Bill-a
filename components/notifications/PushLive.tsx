"use client";
import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";

// A push that arrives while the app is open and visible: the service worker doesn't raise a system notification
// (that would show the same thing twice) and tells the page instead. The bell and an open inbox refresh through the
// same "notifications-changed" event they already use, and a small toast says what happened, with a way in.
export function PushLive({ signedIn }: { signedIn: boolean }) {
  const router = useRouter();
  useEffect(() => {
    if (!signedIn || !("serviceWorker" in navigator)) return;
    const onMessage = (e: MessageEvent) => {
      const d = e.data as { type?: string; id?: string; title?: string; body?: string; url?: string } | null;
      if (!d || d.type !== "push") return;
      window.dispatchEvent(new Event("notifications-changed"));
      const url = typeof d.url === "string" && d.url.startsWith("/") && !d.url.startsWith("//") ? d.url : "/dashboard/notifications";
      const target = typeof d.id === "string" ? `${url}${url.includes("?") ? "&" : "?"}n=${encodeURIComponent(d.id)}` : url;
      toast(d.title || "Bill.a", { description: d.body, action: { label: "Open", onClick: () => router.push(target, { transitionTypes: ["nav-forward"] }) } });
    };
    navigator.serviceWorker.addEventListener("message", onMessage);
    return () => navigator.serviceWorker.removeEventListener("message", onMessage);
  }, [signedIn, router]);
  return null;
}
