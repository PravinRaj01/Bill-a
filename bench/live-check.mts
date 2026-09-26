// Headless live check of the BYOK providers, using the app's real request builders
// against the real APIs. Keys come from the environment (put GROQ_API_KEY and/or
// GEMINI_API_KEY in .env.local — it is gitignored) and are NEVER printed.
//
//   npm run live:check                 key validity + one real plan call per provider
//   npm run live:check -- --enhance    also send a receipt photo to Gemini (Cloud Enhance)
//   npm run live:check -- --bakeoff    run the 15 bake-off + 20 held-out cases per model
//
// It exists because the unit tests mock fetch: this is the only place the first
// SUCCESSFUL call to each provider (schema accepted, answer parseable) is proven.

import fs from "node:fs";
import path from "node:path";
import { BAKEOFF_CASES, type BakeoffCase } from "../lib/ai/bakeoff-cases";
import { HELDOUT_CASES } from "../lib/split/heldout-cases";
import { computeSplit } from "../lib/split/engine";
import { validatePlan } from "../lib/ai/validatePlan";
import { resolveInstruction } from "../lib/retrieval/resolve";
import { groqPlan } from "../lib/ai/providers/groq";
import { geminiPlan } from "../lib/ai/providers/gemini";
import { enhanceReceipt } from "../lib/ai/providers/enhance";
import { testKey } from "../lib/ai/providers/testKey";
import { MODELS } from "../lib/ai/providers/models";
import { ProviderError, type ProviderId } from "../lib/ai/providers/types";

const args = process.argv.slice(2);
const BAKEOFF = args.includes("--bakeoff");
const ENHANCE = args.includes("--enhance");
const keys: Partial<Record<ProviderId, string>> = {
  groq: process.env.GROQ_API_KEY?.trim() || undefined,
  gemini: process.env.GEMINI_API_KEY?.trim() || undefined,
};

const TARGETS: { id: string; provider: ProviderId; model: string }[] = [
  { id: "groq-primary", provider: "groq", model: MODELS.groq.primary },
  { id: "groq-secondary", provider: "groq", model: MODELS.groq.secondary },
  { id: "gemini", provider: "gemini", model: MODELS.gemini.primary },
];
const call = { groq: groqPlan, gemini: geminiPlan } as const;

const describeError = (e: unknown) =>
  e instanceof ProviderError ? `${e.kind}${e.status ? ` (HTTP ${e.status})` : ""}: ${e.message}` : e instanceof Error ? e.message : String(e);

const amounts = (c: BakeoffCase, plan: BakeoffCase["expectedPlan"]) =>
  computeSplit(c.receipt, c.people, plan, c.applyTax).splits.map((s) => s.amount);
const quantile = (xs: number[], q: number) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? s[Math.min(s.length - 1, Math.floor(q * s.length))] : 0;
};

// Free-tier reality (from the response headers): Groq allows 1,000 requests/day but only
// 8,000 tokens/minute (~1,000 per plan call => ~8 calls/min); Gemini flash-lite is ~15
// requests/minute. Space calls out accordingly, and if a 429 still happens, wait as
// long as the provider says and retry.
const SPACING_MS: Record<ProviderId, number> = { groq: 7_500, gemini: 4_500 };
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function planFor(t: (typeof TARGETS)[number], c: BakeoffCase) {
  for (let attempt = 1; ; attempt++) {
    try {
      return await planOnce(t, c);
    } catch (e) {
      if (!(e instanceof ProviderError) || e.kind !== "rate-limit" || attempt >= 4) throw e;
      await sleep(e.retryAfterMs ?? 20_000);
    }
  }
}

async function planOnce(t: (typeof TARGETS)[number], c: BakeoffCase) {
  const started = performance.now();
  const raw = await call[t.provider](
    {
      people: c.people,
      items: c.receipt.items,
      instructions: [c.instruction],
      resolvedBlock: resolveInstruction(c.instruction, c.receipt.items).promptBlock,
    },
    keys[t.provider]!,
    { model: t.model, signal: AbortSignal.timeout(30_000) },
  );
  const ms = performance.now() - started;
  return { ms, raw, checked: validatePlan(raw, { itemCount: c.receipt.items.length, people: c.people }) };
}

async function main() {
  const active = TARGETS.filter((t) => keys[t.provider]);
  console.log(`keys present: groq=${!!keys.groq} gemini=${!!keys.gemini}`);
  if (active.length === 0) {
    console.log("\nNo keys found. Add GROQ_API_KEY and/or GEMINI_API_KEY to .env.local, then run:\n  npm run live:check");
    return;
  }

  // 1. key validity
  for (const p of ["groq", "gemini"] as const) {
    if (keys[p]) console.log(`${p} key test:`, await testKey(p, keys[p]!, { signal: AbortSignal.timeout(15_000) }));
  }

  // 2. one real plan call per target — the moment of truth for the request shapes
  console.log("\n--- one real plan call per model (first successful call) ---");
  const probe = BAKEOFF_CASES.find((c) => c.id === "drinks-payer")!;
  let allOk = true;
  for (const t of active) {
    try {
      const { ms, raw, checked } = await planFor(t, probe);
      const ok = checked.ok;
      allOk &&= ok;
      console.log(`${ok ? "OK  " : "FAIL"} ${t.id.padEnd(10)} ${ms.toFixed(0).padStart(5)} ms  ${ok ? JSON.stringify(checked.plan.assignments) : "invalid plan: " + (checked as { reason: string }).reason + "  raw=" + String(raw).slice(0, 200)}`);
    } catch (e) {
      allOk = false;
      console.log(`FAIL ${t.id.padEnd(10)} ${describeError(e)}`);
    }
  }

  // 3. optional: Cloud Enhance with a real photo
  if (ENHANCE && keys.gemini) {
    console.log("\n--- Cloud Enhance (Gemini vision) ---");
    const img = path.resolve(import.meta.dirname, "receipts", "kaggle-ocr-01.jpg");
    if (!fs.existsSync(img)) console.log("skipped: bench/receipts/kaggle-ocr-01.jpg not found (run bench/prepare-datasets.mjs)");
    else {
      try {
        const started = performance.now();
        const r = await enhanceReceipt(new Blob([fs.readFileSync(img)], { type: "image/jpeg" }), keys.gemini, { signal: AbortSignal.timeout(45_000) });
        console.log(`OK   ${(performance.now() - started).toFixed(0)} ms  items=${r.items.length} total=${(r.receipt.total / 100).toFixed(2)} (true 38.68) tax=${(r.receipt.tax / 100).toFixed(2)} conf=${r.confidence}`);
        for (const w of r.warnings) console.log("     warning:", w);
      } catch (e) {
        allOk = false;
        console.log("FAIL", describeError(e));
      }
    }
  }

  // 4. optional: the full bake-off
  if (BAKEOFF) {
    const cases = [...BAKEOFF_CASES, ...HELDOUT_CASES];
    console.log(`\n--- bake-off: ${cases.length} cases ---`);
    for (const t of active) {
      let correct = 0, wrong = 0, errors = 0;
      const times: number[] = [];
      const misses: string[] = [];
      for (const c of cases) {
        try {
          const { ms, checked } = await planFor(t, c);
          times.push(ms);
          const ok = checked.ok && amounts(c, c.expectedPlan).every((v, i) => v === amounts(c, checked.plan)[i]);
          if (ok) correct++;
          else { wrong++; misses.push(c.id); }
        } catch (e) {
          errors++;
          misses.push(`${c.id} (${describeError(e).slice(0, 60)})`);
        }
        await sleep(SPACING_MS[t.provider]);
      }
      console.log(`${t.id.padEnd(10)} correct ${correct}/${cases.length}  wrong ${wrong}  errors ${errors}  p50 ${quantile(times, 0.5).toFixed(0)} ms  p95 ${quantile(times, 0.95).toFixed(0)} ms`);
      if (misses.length) console.log(`           missed: ${misses.join(", ")}`);
    }
  }

  process.exitCode = allOk ? 0 : 1;
}

main().catch((e) => {
  console.error(describeError(e));
  process.exit(1);
});
