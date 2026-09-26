import type { AssignmentPlan } from "@/types/domain";
import { validatePlan } from "../validatePlan";
import { parseInstruction, type Chip } from "@/lib/split/fallback-parser";
import { geminiPlan } from "./gemini";
import { groqPlan } from "./groq";
import { MODELS } from "./models";
import { ProviderError, type CallContext, type PlanRequest, type ProviderErrorKind, type ProviderId } from "./types";

// Tries the user's Groq key, then their Gemini key, then the deterministic parser.
// It falls through — never throws — on: no key, a rejected key, rate limits, 5xx,
// a retired model, a timeout, being offline, or an answer that fails validatePlan.
// Whatever happens, the caller gets a plan and a record of what was tried, so the
// UI can say "Groq was rate-limited, used Gemini" instead of showing an error.

export interface Attempt {
  tier: ProviderId;
  model?: string;
  ok: boolean;
  kind?: ProviderErrorKind;
  message?: string;
  ms: number;
  /** For a rate limit: how long the provider asked us to wait. */
  retryAfterMs?: number;
}

export interface CascadeResult {
  plan: AssignmentPlan;
  tier: ProviderId | "fallback";
  attempts: Attempt[];
  /** Cleanups validatePlan applied to the winning provider's answer. */
  repairs: string[];
  /** Questions the fallback parser raised (always empty when a cloud tier answered). */
  chips: Chip[];
}

export interface CascadeOptions {
  keys: Partial<Record<ProviderId, string>>;
  /** Overrides DEFAULT_TIMEOUTS for every tier. A hung tier must not stall the whole split. */
  timeoutMs?: number;
  order?: ProviderId[];
  fetchImpl?: typeof fetch;
  /** Override model ids (the bake-off compares several). */
  models?: Partial<Record<ProviderId, string>>;
  /** Try Groq's second model if the first one fails (default on; limits are per model). */
  groqSecondModel?: boolean;
}

/**
 * Per-provider timeouts. Groq answers in well under a second (p95 ~0.8 s measured), so
 * 8 s means it is genuinely stuck. Gemini's free tier measured 3-12 s per call even
 * with thinking off, so it gets 20 s — it is the backup tier, and waiting for a real
 * answer beats timing out into the on-device parser.
 */
export const DEFAULT_TIMEOUTS: Record<ProviderId, number> = { groq: 8000, gemini: 20000 };

const CALLERS: Record<ProviderId, (req: PlanRequest, key: string, ctx: CallContext) => Promise<string>> = {
  groq: groqPlan,
  gemini: geminiPlan,
};

export async function runCascade(req: PlanRequest, opts: CascadeOptions): Promise<CascadeResult> {
  const attempts: Attempt[] = [];
  const itemCount = req.items.length;

  // The ordered list of (provider, model) attempts. Groq gets two: rate limits are per
  // model, so a 429 on the first may not apply to the second. A rejected key would fail
  // both, so once a provider says "auth" it is not tried again.
  const steps: { tier: ProviderId; model?: string }[] = [];
  for (const tier of opts.order ?? (["groq", "gemini"] as ProviderId[])) {
    const chosen = opts.models?.[tier];
    steps.push({ tier, model: chosen ?? (tier === "groq" ? MODELS.groq.primary : undefined) });
    if (tier === "groq" && !chosen && opts.groqSecondModel !== false) {
      steps.push({ tier, model: MODELS.groq.secondary });
    }
  }
  const rejected = new Set<ProviderId>();

  for (const { tier, model } of steps) {
    const key = opts.keys[tier]?.trim();
    if (!key || rejected.has(tier)) continue;

    const started = performance.now();
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), opts.timeoutMs ?? DEFAULT_TIMEOUTS[tier]);
    try {
      const raw = await CALLERS[tier](req, key, { signal: controller.signal, fetchImpl: opts.fetchImpl, model });
      const checked = validatePlan(raw, { itemCount, people: req.people });
      const ms = performance.now() - started;
      if (checked.ok) {
        attempts.push({ tier, model, ok: true, ms });
        return { plan: checked.plan, tier, attempts, repairs: checked.repairs, chips: [] };
      }
      attempts.push({ tier, model, ok: false, kind: "bad-output", message: checked.reason, ms });
    } catch (e) {
      const err =
        e instanceof ProviderError ? e : new ProviderError(tier, "network", e instanceof Error ? e.message : "failed");
      if (err.kind === "auth") rejected.add(tier);
      attempts.push({ tier, model, ok: false, kind: err.kind, message: err.message, ms: performance.now() - started, retryAfterMs: err.retryAfterMs });
    } finally {
      clearTimeout(timer);
    }
  }

  const fallback = parseInstruction(req.instructions.join(". "), req.people, req.items);
  return { plan: fallback.plan, tier: "fallback", attempts, repairs: [], chips: fallback.chips };
}

/** One human sentence about why the cloud tiers didn't answer, or null if none were tried. */
export function explainAttempts(attempts: Attempt[]): string | null {
  const failed = attempts.filter((a) => !a.ok);
  if (failed.length === 0) return null;
  const name = (t: ProviderId) => (t === "groq" ? "Groq" : "Gemini");
  const why = (a: Attempt) =>
    ({
      auth: "rejected your key",
      "rate-limit": "is rate-limited",
      server: "had a server problem",
      "model-gone": "no longer offers this model",
      "bad-request": "rejected the request",
      blocked: "blocked the request",
      "bad-output": "gave an unusable answer",
      timeout: "took too long",
      network: "couldn't be reached",
    })[a.kind ?? "network"];
  // Groq is tried with two models; "Groq is rate-limited; Groq is rate-limited" helps nobody.
  return [...new Set(failed.map((a) => `${name(a.tier)} ${why(a)}`))].join("; ");
}
