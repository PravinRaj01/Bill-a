import { z } from "zod";

// Privacy-first product telemetry: just enough to see whether the AI cascade, the
// OCR and the splitter are healthy in production.
//
// What is NEVER in an event, by construction (the schemas below are `.strict()`, so
// an extra field is rejected, not silently stored): API keys, people's names, item
// names, prices or totals, instructions, receipt text or images, user ids, emails.
// Only categorical facts (which tier answered, an HTTP-ish error kind), coarse
// buckets (OCR confidence in tenths, item-count ranges) and small counters.
//
// Free-text error messages are the one leak risk (a message could echo a key), so they
// are truncated and passed through redact() first.

const tier = z.enum(["groq", "gemini", "fallback"]);
const kind = z.enum([
  "auth", "rate-limit", "server", "model-gone", "bad-request", "blocked", "bad-output", "timeout", "network",
]);

export const telemetryEventSchema = z.discriminatedUnion("type", [
  z
    .object({
      type: z.literal("split"),
      tier,
      /** Which cloud attempts failed on the way, e.g. [{tier:"groq", kind:"rate-limit"}]. */
      attempts: z.array(z.object({ tier: z.enum(["groq", "gemini"]), kind }).strict()).max(6),
      repairs: z.number().int().min(0).max(50),
      ms: z.number().int().min(0).max(300_000),
      /** How many instructions deep the conversation was (1 = first try). */
      rounds: z.number().int().min(1).max(50),
    })
    .strict(),
  z
    .object({
      type: z.literal("scan"),
      engine: z.enum(["tesseract"]),
      /** OCR confidence, in tenths: 0..10. */
      confidence: z.number().int().min(0).max(10),
      items: z.enum(["0", "1-5", "6-15", "16+"]),
      totalSource: z.enum(["keyword", "computed", "max-amount", "none"]),
      warnings: z.number().int().min(0).max(20),
      ms: z.number().int().min(0).max(300_000),
    })
    .strict(),
  z
    .object({
      type: z.literal("crosscheck"),
      /** The cloud tier that was second-guessed. */
      tier: z.enum(["groq", "gemini"]),
      /** shown = the two readings disagreed; ai / rules = which one the user picked. */
      outcome: z.enum(["shown", "ai", "rules"]),
    })
    .strict(),
  z
    .object({
      type: z.literal("enhance"),
      ok: z.boolean(),
      kind: kind.optional(),
      ms: z.number().int().min(0).max(300_000),
    })
    .strict(),
  z
    .object({
      type: z.literal("error"),
      where: z.enum(["split", "scan", "enhance", "render", "sync"]),
      name: z.string().max(60),
      message: z.string().max(160),
      /** SplitReconciliationError: the two cent totals that disagreed (unreachable by design — if it fires, it's a bug). */
      reconciliation: z.object({ expected: z.number().int(), actual: z.number().int() }).strict().optional(),
    })
    .strict(),
]);

export type TelemetryEvent = z.infer<typeof telemetryEventSchema>;

/**
 * Scrubs anything key-shaped out of free text. Deliberately aggressive: a false positive
 * costs a slightly less useful log line, a false negative leaks a credential.
 */
export function redact(text: string): string {
  return text
    .replace(/\bgsk_[A-Za-z0-9_-]{8,}/g, "[key]") //                       Groq
    .replace(/\bAIza[\w-]{8,}/g, "[key]") //                               Google
    .replace(/\b(?:sk|pk|rk)[-_][A-Za-z0-9_-]{16,}/g, "[key]") //          generic sk-… style
    .replace(/\bBearer\s+\S+/gi, "Bearer [token]")
    .replace(/(x-goog-api-key|authorization|api[_-]?key)\s*[:=]\s*\S+/gi, "$1=[redacted]")
    .replace(/[A-Za-z0-9+/_-]{32,}={0,2}/g, "[long-token]") //             any other long opaque string
    .slice(0, 160);
}

export const bucketConfidence = (c: number) => Math.max(0, Math.min(10, Math.round(c * 10)));
export const bucketItems = (n: number): "0" | "1-5" | "6-15" | "16+" =>
  n <= 0 ? "0" : n <= 5 ? "1-5" : n <= 15 ? "6-15" : "16+";
