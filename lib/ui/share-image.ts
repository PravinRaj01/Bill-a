import { domToBlob } from "modern-screenshot";

// Turning the receipt into a picture and sending it. The receipt is drawn as plain HTML at a fixed width
// (components/receipt/ReceiptImage.tsx) and captured here, so what is previewed is exactly what is sent.
//
//   1. navigator.share({ files })  the phone's share sheet with the PNG attached (WhatsApp, Telegram, ...)
//   2. download                    desktop browsers without file sharing: the PNG is saved instead

export type ImageShareOutcome = "shared" | "saved" | "cancelled" | "failed";

// Safari (WebKit) draws every weight as Regular when a web font is embedded in the capture, which flattens the
// whole hierarchy (the total and the amounts lose their bold). There the picture uses the device's own system
// font instead, which bolds correctly; everywhere else it keeps the app's font. The receipt must be LAID OUT
// in that same font (ReceiptImage's `systemFont`): the capture copies each box's measured size, so switching
// fonts only at capture time makes names wrap and overflow.
export const SYSTEM_FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif";
export const SYSTEM_MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";

/** Safari, and every browser on iPhone/iPad (they all run on WebKit, whatever their name). */
export function isWebKit(ua: string): boolean {
  if (/iPhone|iPad|iPod/.test(ua)) return true;
  return /AppleWebKit/.test(ua) && !/Chrome|Chromium|Edg|Android|Firefox/.test(ua);
}

/** The page's fonts must be loaded before capture or the picture falls back to a default font. */
export async function renderPng(node: HTMLElement, opts: { systemFont?: boolean } = {}): Promise<Blob> {
  await document.fonts.ready;
  // Pictures inside the receipt (the portraits) must be decoded or they can come out blank.
  await Promise.all(Array.from(node.querySelectorAll("img")).map((img) => img.decode().catch(() => undefined)));
  const blob = await domToBlob(node, {
    scale: 1,
    type: "image/png",
    backgroundColor: "#0e0f12",
    ...(opts.systemFont ? { font: false as const } : {}),
  });
  if (!blob) throw new Error("The receipt image came back empty");
  return blob;
}

/** True when this browser can send a file through its share sheet. */
export function canShareFile(file: File): boolean {
  return typeof navigator !== "undefined" && typeof navigator.canShare === "function" && typeof navigator.share === "function" && navigator.canShare({ files: [file] });
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

/** A safe file name from the bill title: "Dinner out!" -> "dinner-out-receipt.png". */
export function receiptFilename(title: string): string {
  const slug = title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40);
  return `${slug || "bill"}-receipt.png`;
}

export async function shareImage(blob: Blob, title: string): Promise<ImageShareOutcome> {
  const filename = receiptFilename(title);
  const file = new File([blob], filename, { type: "image/png" });
  if (canShareFile(file)) {
    try {
      await navigator.share({ files: [file], title });
      return "shared";
    } catch (e) {
      // Closing the share sheet is not an error, and must not fall through to a download.
      if (e instanceof DOMException && e.name === "AbortError") return "cancelled";
      // Anything else (not allowed here): save it instead.
    }
  }
  try {
    downloadBlob(blob, filename);
    return "saved";
  } catch {
    return "failed";
  }
}

export function saveImage(blob: Blob, title: string): ImageShareOutcome {
  try {
    downloadBlob(blob, receiptFilename(title));
    return "saved";
  } catch {
    return "failed";
  }
}
