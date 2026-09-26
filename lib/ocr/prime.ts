import { releaseReceiptReaderIfIdle, warmReceiptReader } from "./client";

// "Prime the offline pack": the first time someone opens the new-session page online, load
// the OCR reader once so the service worker caches exactly the files this browser needs
// (which WASM variant it picks depends on the device). After that, scanning works with no
// signal, even from a cold start. Runs once per browser, never on a metered connection.

const FLAG = "billa.ocr.primed.v1";

function shouldSkip(): boolean {
  const conn = (navigator as Navigator & { connection?: { saveData?: boolean; effectiveType?: string } }).connection;
  return !!conn?.saveData || /(^|-)2g$/.test(conn?.effectiveType ?? "");
}

/** Waits (briefly) for the freshly installed worker to take control of this page. */
async function controlled(): Promise<ServiceWorker | null> {
  await navigator.serviceWorker.ready;
  if (navigator.serviceWorker.controller) return navigator.serviceWorker.controller;
  return new Promise((resolve) => {
    const done = () => resolve(navigator.serviceWorker.controller);
    navigator.serviceWorker.addEventListener("controllerchange", done, { once: true });
    setTimeout(done, 3000);
  });
}

export async function primeOfflinePack(): Promise<void> {
  try {
    if (process.env.NODE_ENV !== "production" || !("serviceWorker" in navigator)) return;
    if (localStorage.getItem(FLAG) || shouldSkip() || navigator.onLine === false) return;

    const sw = await controlled();
    if (!sw) return; // first ever load: the next visit will prime

    await warmReceiptReader();
    localStorage.setItem(FLAG, "1");
    sw.postMessage({ type: "cache-page", path: "/dashboard/new" });
    // Don't hold ~100 MB of worker memory for someone who never scans — but only if nobody
    // started using the reader while we were priming it.
    await releaseReceiptReaderIfIdle();
  } catch {
    /* best effort */
  }
}
