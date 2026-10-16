// Light haptic feedback (slider detents, "split done"). Wrapped here because web-haptics'
// iOS path is a Safari quirk that Apple may change: every call is fire-and-forget, never
// throws, and is never the only feedback the user gets.
import type { WebHaptics } from "web-haptics";

type Kind = "selection" | "success" | "error";

let instance: WebHaptics | null | undefined;

async function get(): Promise<WebHaptics | null> {
  if (instance !== undefined) return instance;
  if (typeof window === "undefined") return (instance = null);
  try {
    const { WebHaptics } = await import("web-haptics");
    instance = WebHaptics.isSupported ? new WebHaptics() : null;
  } catch {
    instance = null;
  }
  return instance;
}

const reducedMotion = () =>
  typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

/** Must be called from a user gesture (Android requires sticky user activation). */
export function haptic(kind: Kind = "selection"): void {
  if (reducedMotion()) return;
  void get().then((h) => h?.trigger(kind)).catch(() => {});
}
