"use client";

// DEV-ONLY: measures the split pipeline's accuracy and latency per model, with YOUR
// OWN key (read from the same localStorage the app uses; it never leaves your
// browser except to the provider). Decides which Groq model is the default in
// lib/ai/providers/models.ts, and what the on-device parser is worth.
//
// Scoring is the same as the original bake-off: run the case's expected plan AND the
// model's plan through the real engine (computeSplit) and compare per-person cents —
// so "did it understand the instruction" is separated from arithmetic.
//
// /dev is excluded from auth middleware and must be blocked in production (Phase 7).

import { useEffect, useMemo, useState } from "react";
import { BAKEOFF_CASES, type BakeoffCase } from "@/lib/ai/bakeoff-cases";
import { HELDOUT_CASES } from "@/lib/split/heldout-cases";
import { computeSplit } from "@/lib/split/engine";
import { parseInstruction } from "@/lib/split/fallback-parser";
import { resolveInstruction } from "@/lib/retrieval/resolve";
import { runCascade } from "@/lib/ai/providers/cascade";
import { MODELS } from "@/lib/ai/providers/models";
import type { ProviderId } from "@/lib/ai/providers/types";
import { getKeys, onKeysChanged, type Keys } from "@/lib/ai/keyStore";

type Target =
  | { id: string; label: string; kind: "cloud"; provider: ProviderId; model: string }
  | { id: string; label: string; kind: "fallback" };

const TARGETS: Target[] = [
  { id: "groq-primary", label: `Groq ${MODELS.groq.primary}`, kind: "cloud", provider: "groq", model: MODELS.groq.primary },
  { id: "groq-secondary", label: `Groq ${MODELS.groq.secondary}`, kind: "cloud", provider: "groq", model: MODELS.groq.secondary },
  { id: "gemini", label: `Gemini ${MODELS.gemini.primary}`, kind: "cloud", provider: "gemini", model: MODELS.gemini.primary },
  { id: "fallback", label: "On-device parser (no key)", kind: "fallback" },
];

type Verdict = "correct" | "wrong" | "safe-chip" | "error";
interface Row {
  caseId: string;
  verdict: Verdict;
  ms: number;
  detail: string;
}

const amounts = (c: BakeoffCase, plan: BakeoffCase["expectedPlan"]) =>
  computeSplit(c.receipt, c.people, plan, c.applyTax).splits.map((s) => s.amount);
const sameAmounts = (a: number[], b: number[]) => a.length === b.length && a.every((v, i) => v === b[i]);
const quantile = (xs: number[], q: number) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? s[Math.min(s.length - 1, Math.floor(q * s.length))] : 0;
};

async function runCase(t: Target, c: BakeoffCase, keys: Keys, withResolution: boolean, tries = 0): Promise<Row> {
  const started = performance.now();
  try {
    let plan: BakeoffCase["expectedPlan"];
    let chips = 0;
    if (t.kind === "fallback") {
      const r = parseInstruction(c.instruction, c.people, c.receipt.items);
      plan = r.plan;
      chips = r.chips.length;
    } else {
      const resolvedBlock = withResolution ? resolveInstruction(c.instruction, c.receipt.items).promptBlock : undefined;
      const r = await runCascade(
        { people: c.people, items: c.receipt.items, instructions: [c.instruction], resolvedBlock },
        { keys: { [t.provider]: keys[t.provider] }, order: [t.provider], models: { [t.provider]: t.model }, timeoutMs: 30_000 },
      );
      if (r.tier === "fallback") {
        const a = r.attempts[0];
        // A 429 is the provider's free-tier limit, not a wrong answer: wait as long as it
        // says (default 20 s) and try this case again, up to 3 more times.
        if (a?.kind === "rate-limit" && tries < 3) {
          await new Promise((res) => setTimeout(res, a.retryAfterMs ?? 20_000));
          return runCase(t, c, keys, withResolution, tries + 1);
        }
        return { caseId: c.id, verdict: "error", ms: performance.now() - started, detail: `${a?.kind ?? "no key"}: ${a?.message ?? ""}` };
      }
      plan = r.plan;
    }
    const ms = performance.now() - started;
    const ok = sameAmounts(amounts(c, c.expectedPlan), amounts(c, plan));
    if (ok) return { caseId: c.id, verdict: "correct", ms, detail: "" };
    return { caseId: c.id, verdict: chips > 0 ? "safe-chip" : "wrong", ms, detail: JSON.stringify(plan.assignments) };
  } catch (e) {
    return { caseId: c.id, verdict: "error", ms: performance.now() - started, detail: e instanceof Error ? e.message : String(e) };
  }
}

export default function ModelBakeoff() {
  const [keys, setKeys] = useState<Keys>({});
  const [chosen, setChosen] = useState<Record<string, boolean>>({ "groq-primary": true, "groq-secondary": true, gemini: true, fallback: true });
  const [set, setSet] = useState<"dev" | "heldout" | "both">("dev");
  const [withResolution, setWithResolution] = useState(true);
  const [rows, setRows] = useState<Record<string, Row[]>>({});
  const [running, setRunning] = useState<string | null>(null);

  useEffect(() => {
    setKeys(getKeys());
    return onKeysChanged(() => setKeys(getKeys()));
  }, []);

  const cases = useMemo(
    () => (set === "dev" ? BAKEOFF_CASES : set === "heldout" ? HELDOUT_CASES : [...BAKEOFF_CASES, ...HELDOUT_CASES]),
    [set],
  );

  const run = async () => {
    setRows({});
    for (const t of TARGETS.filter((x) => chosen[x.id])) {
      if (t.kind === "cloud" && !keys[t.provider]) {
        setRows((r) => ({ ...r, [t.id]: [{ caseId: "-", verdict: "error", ms: 0, detail: `no ${t.provider} key saved (open the app → AI settings)` }] }));
        continue;
      }
      setRunning(t.label);
      const out: Row[] = [];
      for (const c of cases) {
        out.push(await runCase(t, c, keys, withResolution));
        setRows((r) => ({ ...r, [t.id]: [...out] }));
        // Free tier: Groq ~8,000 tokens/min (~8 plan calls/min), Gemini flash-lite ~15 requests/min.
        if (t.kind === "cloud") await new Promise((res) => setTimeout(res, t.provider === "groq" ? 7500 : 4500));
      }
    }
    setRunning(null);
  };

  const summary = TARGETS.filter((t) => rows[t.id]).map((t) => {
    const r = rows[t.id];
    const ms = r.filter((x) => x.verdict !== "error").map((x) => x.ms);
    const n = (v: Verdict) => r.filter((x) => x.verdict === v).length;
    return { t, n: r.length, correct: n("correct"), wrong: n("wrong"), chip: n("safe-chip"), error: n("error"), p50: quantile(ms, 0.5), p95: quantile(ms, 0.95) };
  });

  const markdown = () =>
    [
      `# Split bake-off (${new Date().toISOString().slice(0, 10)})`,
      "",
      `Cases: ${set} (${cases.length}) · reference resolution in prompt: ${withResolution ? "yes" : "no"}`,
      "",
      "| target | correct | wrong | asked (chip) | error | p50 ms | p95 ms |",
      "|---|---|---|---|---|---|---|",
      ...summary.map((s) => `| ${s.t.label} | ${s.correct}/${s.n} | ${s.wrong} | ${s.chip} | ${s.error} | ${s.p50.toFixed(0)} | ${s.p95.toFixed(0)} |`),
      "",
      ...summary.flatMap((s) => [
        `## ${s.t.label}`,
        ...rows[s.t.id].filter((r) => r.verdict !== "correct").map((r) => `- ${r.caseId}: ${r.verdict}${r.detail ? ` — ${r.detail}` : ""}`),
        "",
      ]),
    ].join("\n");

  const download = () => {
    const url = URL.createObjectURL(new Blob([markdown()], { type: "text/markdown" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = "split-bakeoff.md";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <main className="mx-auto max-w-4xl space-y-6 bg-black p-6 font-mono text-sm text-zinc-200">
      <h1 className="text-xl font-bold text-white">Split bake-off (cloud models + on-device parser)</h1>
      <p className="text-xs text-zinc-500">
        Uses the keys saved in this browser (AI settings in the app). Calls go straight to the provider. Groq key: {keys.groq ? "saved" : "missing"} · Gemini key: {keys.gemini ? "saved" : "missing"}.
      </p>

      <div className="flex flex-wrap items-center gap-4 rounded border border-white/10 p-4">
        {TARGETS.map((t) => (
          <label key={t.id} className="flex items-center gap-2">
            <input type="checkbox" checked={!!chosen[t.id]} onChange={(e) => setChosen((c) => ({ ...c, [t.id]: e.target.checked }))} />
            {t.label}
          </label>
        ))}
        <select value={set} onChange={(e) => setSet(e.target.value as typeof set)} className="bg-black border border-white/20 p-1">
          <option value="dev">15 bake-off cases</option>
          <option value="heldout">20 held-out cases</option>
          <option value="both">both (35)</option>
        </select>
        <label className="flex items-center gap-2">
          <input type="checkbox" checked={withResolution} onChange={(e) => setWithResolution(e.target.checked)} />
          resolved references in prompt
        </label>
        <button onClick={run} disabled={!!running} className="rounded bg-white px-4 py-1 font-bold text-black disabled:opacity-40">
          {running ? `Running ${running}…` : "Run"}
        </button>
        {summary.length > 0 && !running && (
          <button onClick={download} className="rounded border border-white/30 px-4 py-1">Download report</button>
        )}
      </div>

      {summary.length > 0 && (
        <table className="w-full border-collapse text-left text-xs">
          <thead>
            <tr className="border-b border-white/20 text-zinc-500">
              <th className="py-2">target</th><th>correct</th><th>wrong</th><th>asked</th><th>error</th><th>p50 ms</th><th>p95 ms</th>
            </tr>
          </thead>
          <tbody>
            {summary.map((s) => (
              <tr key={s.t.id} className="border-b border-white/5">
                <td className="py-2">{s.t.label}</td>
                <td className="text-emerald-400">{s.correct}/{s.n}</td>
                <td className="text-red-400">{s.wrong}</td>
                <td className="text-amber-300">{s.chip}</td>
                <td className="text-red-400">{s.error}</td>
                <td>{s.p50.toFixed(0)}</td>
                <td>{s.p95.toFixed(0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {summary.map((s) => (
        <details key={s.t.id} className="rounded border border-white/10 p-3">
          <summary className="cursor-pointer">{s.t.label} — per case</summary>
          <ul className="mt-2 space-y-1 text-xs">
            {rows[s.t.id].map((r, i) => (
              <li key={i} className={r.verdict === "correct" ? "text-zinc-500" : r.verdict === "safe-chip" ? "text-amber-300" : "text-red-400"}>
                {r.caseId}: {r.verdict} ({r.ms.toFixed(0)} ms){r.detail ? ` — ${r.detail}` : ""}
              </li>
            ))}
          </ul>
        </details>
      ))}
    </main>
  );
}
