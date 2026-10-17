// Sharing a message ("Riley owes RM41.20", a nudge): the phone's own share sheet first, so the person
// picks WhatsApp, Telegram, SMS, whatever they use. The old `whatsapp://` link only worked on phones
// with WhatsApp installed and did nothing on a desktop.
//
//   1. navigator.share   (phones, tablets, Windows/macOS Chrome and Safari)
//   2. wa.me link        (desktop browsers without the share sheet: opens WhatsApp Web)
//   3. clipboard         (last resort; the caller tells the user it was copied)

import { waLink } from "@/lib/people/phone";

export type ShareOutcome = "shared" | "opened" | "copied" | "cancelled" | "failed";

export async function shareText(text: string): Promise<ShareOutcome> {
  if (typeof navigator !== "undefined" && typeof navigator.share === "function") {
    try {
      await navigator.share({ text });
      return "shared";
    } catch (e) {
      // Closing the share sheet is not an error, and must not fall through to opening another window.
      if (e instanceof DOMException && e.name === "AbortError") return "cancelled";
      // Anything else (not allowed here, unsupported data): try the next way.
    }
  }

  if (typeof window !== "undefined") {
    const win = window.open(`https://wa.me/?text=${encodeURIComponent(text)}`, "_blank", "noopener,noreferrer");
    if (win) return "opened";
  }

  try {
    await navigator.clipboard.writeText(text);
    return "copied";
  } catch {
    return "failed";
  }
}

/**
 * A message to one person. With their WhatsApp number it opens that chat directly (wa.me/<number>), so the
 * nudge goes to the right person without picking from the share sheet; without one it's a normal share.
 */
export async function shareToPerson(text: string, phone?: string | null): Promise<ShareOutcome> {
  if (phone && typeof window !== "undefined") {
    const win = window.open(waLink(phone, text), "_blank", "noopener,noreferrer");
    if (win) return "opened";
  }
  return shareText(text);
}

/** The toast line for an outcome, or null when nothing needs saying (the share sheet speaks for itself). */
export function shareMessage(o: ShareOutcome): string | null {
  if (o === "copied") return "Message copied. Paste it into your chat.";
  if (o === "failed") return "Couldn't share that. Try again.";
  return null;
}
