import { redact, type TelemetryEvent } from "./events";

// Fire-and-forget: telemetry must never slow the app down or break it. It does nothing
// in development, respects Do Not Track, and swallows every failure.

const ENDPOINT = "/api/telemetry";

function enabled(): boolean {
  if (process.env.NODE_ENV !== "production") return false;
  if (typeof navigator === "undefined") return false;
  return navigator.doNotTrack !== "1" && (window as { doNotTrack?: string }).doNotTrack !== "1";
}

export function track(event: TelemetryEvent): void {
  try {
    if (!enabled()) return;
    const body = JSON.stringify(event);
    // sendBeacon survives page unloads; fetch(keepalive) is the fallback.
    if (navigator.sendBeacon?.(ENDPOINT, new Blob([body], { type: "application/json" }))) return;
    void fetch(ENDPOINT, { method: "POST", body, headers: { "Content-Type": "application/json" }, keepalive: true }).catch(() => {});
  } catch {
    /* telemetry is optional */
  }
}

/** Turns anything thrown into a safe, redacted error event. */
export function trackError(where: Extract<TelemetryEvent, { type: "error" }>["where"], err: unknown): void {
  const e = err as { name?: unknown; message?: unknown; expected?: unknown; actual?: unknown };
  const reconciliation =
    e?.name === "SplitReconciliationError" && Number.isInteger(e.expected) && Number.isInteger(e.actual)
      ? { expected: e.expected as number, actual: e.actual as number }
      : undefined;
  track({
    type: "error",
    where,
    name: redact(typeof e?.name === "string" ? e.name : "Error").slice(0, 60),
    // For a reconciliation error the numbers are already in the structured field; the
    // prose is dropped so nothing but cents can reach the log.
    message: reconciliation ? "reconciliation failed" : redact(typeof e?.message === "string" ? e.message : String(err)),
    ...(reconciliation ? { reconciliation } : {}),
  });
}
