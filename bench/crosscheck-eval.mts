// Measures the cross-check on REAL model output: for every bake-off + held-out case, ask the
// cloud model (through the real planSplit, cross-check on) and classify what the user would see.
//
//   caught     the model was WRONG and the cross-check flagged it     (the whole point)
//   missed     the model was WRONG and nothing was flagged            (a wrong split goes through)
//   agreed     the model was right and nothing was flagged            (the normal, free case)
//   false      the model was RIGHT but the rules disagreed anyway     (a needless prompt)
//   asked      the rules/router asked a question instead of running   (not a model outcome)
//
// Keys come from .env.local; nothing is printed but counts and case ids.
//   npm run crosscheck:eval

import { BAKEOFF_CASES, type BakeoffCase } from "../lib/ai/bakeoff-cases";
import { HELDOUT_CASES } from "../lib/split/heldout-cases";
import { computeSplit } from "../lib/split/engine";
import { planSplit } from "../lib/ai/planSplit";
import type { ProviderId } from "../lib/ai/providers/types";

const keys: Partial<Record<ProviderId, string>> = { groq: process.env.GROQ_API_KEY, gemini: process.env.GEMINI_API_KEY };
const amounts = (c: BakeoffCase, plan: BakeoffCase["expectedPlan"]) => computeSplit(c.receipt, c.people, plan, c.applyTax).splits.map((s) => s.amount);
const same = (a: number[], b: number[]) => a.length === b.length && a.every((v, i) => v === b[i]);
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const SPACING: Record<ProviderId, number> = { groq: 8_000, gemini: 5_000 };

const cases: (BakeoffCase & { mustAsk?: boolean })[] = [...BAKEOFF_CASES, ...HELDOUT_CASES];

for (const provider of ["groq", "gemini"] as ProviderId[]) {
  if (!keys[provider]) continue;
  const tally = { caught: [] as string[], missed: [] as string[], agreed: 0, falseAlarm: [] as string[], asked: [] as string[], errors: 0 };
  for (const c of cases) {
    const want = amounts(c, c.expectedPlan);
    try {
      const o = await planSplit({
        receipt: c.receipt,
        people: c.people,
        instructions: [c.instruction],
        applyTax: c.applyTax,
        keys: { [provider]: keys[provider] },
        ignoreAmbiguities: true, // isolate the model + cross-check from the typo/unknown-name gate
        timeoutMs: 30_000,
      });
      if (o.kind === "disagreement") {
        const aiRight = same(o.ai.result.splits.map((s) => s.amount), want);
        (aiRight ? tally.falseAlarm : tally.caught).push(c.id);
      } else if (o.kind === "split") {
        if (o.tier === "fallback") tally.errors++;
        else if (same(o.result.splits.map((s) => s.amount), want)) tally.agreed++;
        else tally.missed.push(c.id);
      } else {
        tally.asked.push(c.id);
      }
    } catch {
      tally.errors++;
    }
    await sleep(SPACING[provider]);
  }
  const wrong = tally.caught.length + tally.missed.length;
  console.log(`\n== ${provider}  (${cases.length} cases)`);
  console.log(`  model wrong on ${wrong}: caught ${tally.caught.length}, missed ${tally.missed.length}`);
  console.log(`  model right and quiet: ${tally.agreed}; needless prompts (false alarms): ${tally.falseAlarm.length}; questions asked: ${tally.asked.length}; errors/fallbacks: ${tally.errors}`);
  if (tally.caught.length) console.log("  caught:", tally.caught.join(", "));
  if (tally.missed.length) console.log("  MISSED:", tally.missed.join(", "));
  if (tally.falseAlarm.length) console.log("  false alarms:", tally.falseAlarm.join(", "));
}
