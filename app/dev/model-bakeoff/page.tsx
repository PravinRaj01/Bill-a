"use client";

// Phase 1 model bake-off — plan §3.2. Not product UI.
//
// Runs each candidate model against the fixed case set in
// lib/ai/bakeoff-cases.ts and scores it by running BOTH the model's
// AssignmentPlan and the hand-authored expected AssignmentPlan through
// the same, already-tested computeSplit() engine, then comparing the
// resulting per-person cent amounts. This isolates "did the model
// understand the instruction" from "was the arithmetic right" — the
// arithmetic is proven separately by lib/split/engine.test.ts.
//
// Results persist to localStorage per model id, so running all
// candidates (each requiring a full model download + page reuse) builds
// up a leaderboard without manual note-taking. Wrapped in try/catch per
// the usual caveat: localStorage can throw or silently no-op in private
// browsing, and that must never break the harness itself.

import { useCallback, useEffect, useRef, useState } from "react";
import { createEngine, isModelCached } from "@/lib/ai/engine-client";
import { buildAssignmentPlanSchema } from "@/lib/ai/schemas";
import { buildSystemPrompt, buildUserPrompt, buildItemMenu } from "@/lib/ai/prompts";
import { resolveInstruction } from "@/lib/retrieval/resolve";
import { BAKEOFF_CASES, type BakeoffCase } from "@/lib/ai/bakeoff-cases";
import { computeSplit } from "@/lib/split/engine";
import type { AssignmentPlan, SplitRecord } from "@/types/domain";
import type { InitProgressReport, MLCEngineInterface } from "@mlc-ai/web-llm";

// Run 1 (original 5, kept for the leaderboard's historical record):
// gemma3-1b-it and Qwen3-0.6B both hit WebLLM's documented "spins on
// whitespace until max_tokens" failure — not a capability gap, a stall.
// Qwen2.5-0.5B returned near-constant empty assignments regardless of
// instruction complexity, i.e. it wasn't engaging with the task at all.
// Only Llama-3.2-1B and Qwen2.5-1.5B showed real capability, and their
// failures clustered into two fixable patterns (see lib/ai/prompts.ts's
// revision history comment) rather than random noise.
//
// Run 2 additions: the original plan (§3.2) rejected these as
// "over-specced" under the assumption that an assignment-only task needs
// little capability. Run 1's results partially contradict that — even
// the capable models struggled with categorical binding independent of
// the weights bug — so they're back in for a real comparison instead of
// an assumption. All three are upstream-validated for JSON-schema mode
// in WebLLM's own examples/json-schema.
const CANDIDATES = [
  "gemma3-1b-it-q4f16_1-MLC",
  "Llama-3.2-1B-Instruct-q4f16_1-MLC",
  "Qwen2.5-0.5B-Instruct-q4f16_1-MLC",
  "Qwen3-0.6B-q4f16_1-MLC",
  "Qwen2.5-1.5B-Instruct-q4f16_1-MLC", // optional upgrade tier, plan §3.2
  "Llama-3.2-3B-Instruct-q4f16_1-MLC", // run 2: previously rejected as over-specced
  "Phi-3.5-mini-instruct-q4f16_1-MLC", // run 2: previously rejected as over-specced
  "gemma-2-2b-it-q4f16_1-MLC", // run 2: gemma3-1b's failure was config-specific; try the gemma-2 family
] as const;

interface CaseResult {
  caseId: string;
  pass: boolean;
  reason?: string; // "parse_error" | "split_error" | "mismatch"
  elapsedMs: number;
  actualPlan?: AssignmentPlan;
  actualSplits?: SplitRecord[];
  expectedSplits?: SplitRecord[];
  rawOutput?: string;
  /** What lib/retrieval/resolve.ts injected into the prompt, if anything — kept for debugging. */
  resolvedBlock?: string;
}

interface ModelRunResult {
  modelId: string;
  usedResolution: boolean;
  accuracy: number; // 0-1
  avgElapsedMs: number;
  timestamp: number;
  cases: CaseResult[];
}

const STORAGE_PREFIX = "billa_bakeoff_";

// Keyed by modelId + variant (not just modelId) so a "with resolution"
// run never overwrites the "without" run for the same model — the
// decision gate in plan §11 needs BOTH numbers side by side to mean
// anything ("same models, same cases, nothing else changed").
const storageKey = (modelId: string, usedResolution: boolean) =>
  `${STORAGE_PREFIX}${modelId}${usedResolution ? "__resolved" : "__raw"}`;

function loadLeaderboard(): Record<string, ModelRunResult> {
  const out: Record<string, ModelRunResult> = {};
  try {
    for (const id of CANDIDATES) {
      for (const variant of [false, true]) {
        const raw = localStorage.getItem(storageKey(id, variant));
        if (raw) out[storageKey(id, variant)] = JSON.parse(raw);
      }
    }
  } catch {
    // Private browsing / storage disabled — harness still works, just
    // without cross-run persistence.
  }
  return out;
}

/**
 * Builds a self-contained markdown report from every persisted model run
 * so results can be handed off (as a file) rather than relayed one
 * screenshot at a time. Includes full expected/actual splits and raw
 * model output for every failing case, and a one-line summary for
 * passes — enough to diagnose *why* a model failed without needing to
 * re-run anything.
 */
function generateReport(leaderboard: Record<string, ModelRunResult>): string {
  const lines: string[] = [];
  lines.push("# Bill.a Model Bake-off Report");
  lines.push(`Generated: ${new Date().toISOString()}`);
  lines.push(`Case set: ${BAKEOFF_CASES.length} cases from lib/ai/bakeoff-cases.ts`);
  lines.push("");
  lines.push(
    "Scoring: both the model's AssignmentPlan and the hand-authored expectedPlan are run " +
      "through the real computeSplit() engine; a case passes only if every person's final " +
      "amount matches exactly. This isolates instruction-following from arithmetic.",
  );
  lines.push("");

  const keysFor = (id: string) => [storageKey(id, false), storageKey(id, true)];
  const run = CANDIDATES.flatMap((id) => keysFor(id))
    .map((k) => leaderboard[k])
    .filter((r): r is ModelRunResult => !!r);

  lines.push("## Leaderboard");
  lines.push("");
  lines.push(
    "Mode: **raw** = today's prompt, model searches the item list itself. **resolved** = " +
      "plan §5.1b's deterministic resolution layer injects pre-resolved indices; nothing else " +
      "changes (same model, same cases, same schema). This is the decision-gate comparison.",
  );
  lines.push("");
  lines.push("| Model | Mode | Accuracy | Pass/Total | Avg latency | Run at |");
  lines.push("|---|---|---|---|---|---|");
  for (const id of CANDIDATES) {
    for (const variant of [false, true]) {
      const r = leaderboard[storageKey(id, variant)];
      const mode = variant ? "resolved" : "raw";
      if (!r) {
        lines.push(`| ${id} | ${mode} | — | not run | — | — |`);
        continue;
      }
      const passCount = r.cases.filter((c) => c.pass).length;
      lines.push(
        `| ${id} | ${mode} | ${(r.accuracy * 100).toFixed(0)}% | ${passCount}/${r.cases.length} | ` +
          `${r.avgElapsedMs.toFixed(0)}ms | ${new Date(r.timestamp).toLocaleString()} |`,
      );
    }
  }
  lines.push("");

  for (const r of run) {
    lines.push(`## ${r.modelId} (${r.usedResolution ? "resolved" : "raw"})`);
    lines.push("");
    const passCount = r.cases.filter((c) => c.pass).length;
    lines.push(
      `Accuracy: ${(r.accuracy * 100).toFixed(0)}% (${passCount}/${r.cases.length}), ` +
        `avg latency ${r.avgElapsedMs.toFixed(0)}ms`,
    );
    lines.push("");

    for (const cr of r.cases) {
      const c = BAKEOFF_CASES.find((bc) => bc.id === cr.caseId);
      const mark = cr.pass ? "✓" : "✗";
      lines.push(`### ${mark} ${cr.caseId} — "${c?.instruction ?? "(unknown case)"}"`);
      lines.push("");
      lines.push(`- Elapsed: ${cr.elapsedMs.toFixed(0)}ms`);
      if (cr.reason) lines.push(`- Reason: ${cr.reason}`);
      if (cr.resolvedBlock) {
        lines.push("Resolved references injected into the prompt:");
        lines.push("```");
        lines.push(cr.resolvedBlock);
        lines.push("```");
      }
      if (!cr.pass) {
        lines.push("");
        lines.push("Expected splits:");
        lines.push("```json");
        lines.push(JSON.stringify(cr.expectedSplits ?? "n/a", null, 2));
        lines.push("```");
        lines.push("Actual splits:");
        lines.push("```json");
        lines.push(JSON.stringify(cr.actualSplits ?? "n/a", null, 2));
        lines.push("```");
        if (cr.actualPlan) {
          lines.push("Actual plan:");
          lines.push("```json");
          lines.push(JSON.stringify(cr.actualPlan, null, 2));
          lines.push("```");
        }
        lines.push("Raw model output:");
        lines.push("```");
        lines.push(cr.rawOutput ?? "n/a");
        lines.push("```");
      }
      lines.push("");
    }
  }

  return lines.join("\n");
}

function downloadReport(leaderboard: Record<string, ModelRunResult>) {
  const markdown = generateReport(leaderboard);
  const blob = new Blob([markdown], { type: "text/markdown" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `billa-bakeoff-report-${Date.now()}.md`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function saveResult(result: ModelRunResult) {
  try {
    localStorage.setItem(storageKey(result.modelId, result.usedResolution), JSON.stringify(result));
  } catch {
    // Non-fatal — see loadLeaderboard.
  }
}

function splitsEqual(a: SplitRecord[], b: SplitRecord[]): boolean {
  if (a.length !== b.length) return false;
  const byName = new Map(b.map((s) => [s.name, s.amount]));
  return a.every((s) => byName.get(s.name) === s.amount);
}

function runCase(
  engine: MLCEngineInterface,
  c: BakeoffCase,
  useResolution: boolean,
): Promise<CaseResult> {
  const candidateIndices = c.receipt.items.map((_, i) => i);
  const schema = buildAssignmentPlanSchema(candidateIndices, c.people);

  const menu = buildItemMenu(c.receipt.items);

  // Plan §5.1b / decision gate: this is the ONLY thing that changes
  // between "round 3" (no resolution) and "round 4" (with it) numbers —
  // same models, same cases, same schema, same everything else. That's
  // what makes the before/after comparison mean something.
  const resolution = useResolution ? resolveInstruction(c.instruction, c.receipt.items) : null;

  const messages = [
    { role: "system" as const, content: buildSystemPrompt(candidateIndices, c.people) },
    {
      role: "user" as const,
      content: buildUserPrompt(c.people, menu, c.instruction, resolution?.promptBlock),
    },
  ];

  const expectedSplits = computeSplit(c.receipt, c.people, c.expectedPlan, c.applyTax).splits;

  const t0 = performance.now();
  return engine.chat.completions
    .create({
      messages,
      temperature: 0,
      stream: false,
      max_tokens: 512,
      response_format: { type: "json_object", schema: JSON.stringify(schema) },
    })
    .then((response) => {
      const elapsedMs = performance.now() - t0;
      const raw = response.choices[0]?.message?.content ?? "";

      const resolvedBlock = resolution?.promptBlock || undefined;

      let plan: AssignmentPlan;
      try {
        plan = JSON.parse(raw) as AssignmentPlan;
      } catch {
        return { caseId: c.id, pass: false, reason: "parse_error", elapsedMs, rawOutput: raw, resolvedBlock };
      }

      try {
        const actualSplits = computeSplit(c.receipt, c.people, plan, c.applyTax).splits;
        const pass = splitsEqual(actualSplits, expectedSplits);
        return {
          caseId: c.id,
          pass,
          reason: pass ? undefined : "mismatch",
          elapsedMs,
          actualPlan: plan,
          actualSplits,
          expectedSplits,
          rawOutput: raw,
          resolvedBlock,
        };
      } catch {
        // Should be unreachable — computeSplit's own defensive dedup
        // handles duplicate itemIndex / duplicate names — but a model
        // could still name an out-of-range index if it ever fell back to
        // freeform generation, so this stays a genuine failure mode to
        // record rather than let crash the whole harness run.
        return { caseId: c.id, pass: false, reason: "split_error", elapsedMs, rawOutput: raw, resolvedBlock };
      }
    });
}

export default function ModelBakeoffPage() {
  const [modelId, setModelId] = useState<string>(CANDIDATES[0]);
  const [engine, setEngine] = useState<MLCEngineInterface | null>(null);
  // Not rendered — held so loadModel can dispose of the PREVIOUS engine
  // before creating a new one. Forgetting this is what caused the
  // 20-30x latency blowup on later-loaded models in the first re-run
  // (GPU resources from every earlier model's Worker never released).
  const disposeRef = useRef<(() => Promise<void>) | null>(null);
  const [progress, setProgress] = useState("");
  const [wasCached, setWasCached] = useState<boolean | null>(null);
  const [running, setRunning] = useState(false);
  const [liveIndex, setLiveIndex] = useState(0);
  const [results, setResults] = useState<CaseResult[]>([]);
  const [leaderboard, setLeaderboard] = useState<Record<string, ModelRunResult>>({});
  const [error, setError] = useState("");
  // Default true: this is Phase 2a's whole point. Untick to reproduce a
  // "raw" (round 3) number for direct comparison — same model, same
  // cases, only this changes.
  const [useResolution, setUseResolution] = useState(true);

  useEffect(() => {
    setLeaderboard(loadLeaderboard());
  }, []);

  // Release GPU resources if the user navigates away mid-session instead
  // of clicking "Load model" again (which is the other dispose path).
  useEffect(() => {
    return () => {
      disposeRef.current?.();
    };
  }, []);

  const loadModel = useCallback(async () => {
    setError("");
    setEngine(null);
    setResults([]);
    try {
      if (disposeRef.current) {
        setProgress("Releasing previous model's GPU resources...");
        await disposeRef.current();
        disposeRef.current = null;
      }
      const cached = await isModelCached(modelId);
      setWasCached(cached);
      const { engine: eng, dispose } = await createEngine(modelId, (r: InitProgressReport) =>
        setProgress(r.text),
      );
      disposeRef.current = dispose;
      setEngine(eng);
    } catch (e) {
      setError(`Load failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  }, [modelId]);

  const runAll = useCallback(async () => {
    if (!engine) return;
    setRunning(true);
    setError("");
    const collected: CaseResult[] = [];

    for (let i = 0; i < BAKEOFF_CASES.length; i++) {
      setLiveIndex(i);
      try {
        const r = await runCase(engine, BAKEOFF_CASES[i], useResolution);
        collected.push(r);
        setResults([...collected]);
      } catch (e) {
        collected.push({
          caseId: BAKEOFF_CASES[i].id,
          pass: false,
          reason: "engine_error",
          elapsedMs: 0,
        });
        setResults([...collected]);
      }
    }

    const passCount = collected.filter((r) => r.pass).length;
    const summary: ModelRunResult = {
      modelId,
      usedResolution: useResolution,
      accuracy: passCount / collected.length,
      avgElapsedMs: collected.reduce((s, r) => s + r.elapsedMs, 0) / collected.length,
      timestamp: Date.now(),
      cases: collected,
    };
    saveResult(summary);
    setLeaderboard(loadLeaderboard());
    setRunning(false);
  }, [engine, modelId, useResolution]);

  return (
    <main style={{ maxWidth: 900, margin: "0 auto", padding: 24, fontFamily: "monospace", fontSize: 13 }}>
      <h1 style={{ fontSize: 18, fontWeight: 700 }}>Model Bake-off (Phase 1 — dev only)</h1>
      <p style={{ opacity: 0.7 }}>
        {BAKEOFF_CASES.length} cases. Scores by running both the model's plan and the hand-authored
        expected plan through the real computeSplit() engine and comparing final amounts — not by
        diffing JSON structure. Not product UI.
      </p>

      {/* Leaderboard */}
      <section style={{ marginTop: 16 }}>
        <h2 style={{ fontSize: 14, fontWeight: 700 }}>Leaderboard (this browser)</h2>
        <table style={{ width: "100%", borderCollapse: "collapse", marginTop: 8 }}>
          <thead>
            <tr style={{ textAlign: "left", borderBottom: "1px solid #555" }}>
              <th style={{ padding: 4 }}>Model</th>
              <th style={{ padding: 4 }}>Mode</th>
              <th style={{ padding: 4 }}>Accuracy</th>
              <th style={{ padding: 4 }}>Avg latency</th>
              <th style={{ padding: 4 }}>Run at</th>
            </tr>
          </thead>
          <tbody>
            {CANDIDATES.flatMap((id) =>
              [false, true].map((variant) => {
                const key = storageKey(id, variant);
                const r = leaderboard[key];
                return (
                  <tr key={key} style={{ borderBottom: "1px solid #333" }}>
                    <td style={{ padding: 4 }}>{id}</td>
                    <td style={{ padding: 4, opacity: 0.7 }}>{variant ? "resolved" : "raw"}</td>
                    <td style={{ padding: 4, color: r && r.accuracy >= 0.9 ? "#0f0" : r ? "#fc0" : undefined }}>
                      {r ? `${(r.accuracy * 100).toFixed(0)}% (${r.cases.filter((c) => c.pass).length}/${r.cases.length})` : "—"}
                    </td>
                    <td style={{ padding: 4 }}>{r ? `${r.avgElapsedMs.toFixed(0)}ms` : "—"}</td>
                    <td style={{ padding: 4 }}>{r ? new Date(r.timestamp).toLocaleTimeString() : "—"}</td>
                  </tr>
                );
              }),
            )}
          </tbody>
        </table>
        <p style={{ opacity: 0.5, marginTop: 4 }}>
          Decision gate (plan §11): ≥13/15 resolved → ship, skip fine-tuning. 10-12/15 → train the
          residual. &lt;10/15 → the resolution layer needs rework first.
        </p>
        <button
          onClick={() => downloadReport(leaderboard)}
          disabled={Object.keys(leaderboard).length === 0}
          style={{ marginTop: 8 }}
        >
          Download full report (.md)
        </button>{" "}
        <span style={{ opacity: 0.5 }}>
          Saves every run's full expected/actual splits and raw model output — hand the file off
          instead of screenshotting individual cases.
        </span>
      </section>

      {/* Controls */}
      <section style={{ marginTop: 24 }}>
        <select
          value={modelId}
          onChange={(e) => setModelId(e.target.value)}
          disabled={running}
          style={{ background: "#111", color: "#0f0", padding: 4 }}
        >
          {CANDIDATES.map((id) => (
            <option key={id} value={id}>
              {id}
            </option>
          ))}
        </select>{" "}
        <button onClick={loadModel} disabled={running}>
          Load model
        </button>{" "}
        <button onClick={runAll} disabled={!engine || running}>
          Run all {BAKEOFF_CASES.length} cases
        </button>{" "}
        <label style={{ marginLeft: 8 }}>
          <input
            type="checkbox"
            checked={useResolution}
            onChange={(e) => setUseResolution(e.target.checked)}
            disabled={running}
          />{" "}
          Use resolution layer (§5.1b) — untick to reproduce a raw/round-3 number
        </label>
        {wasCached !== null && <p style={{ marginTop: 8 }}>Was already cached: {String(wasCached)}</p>}
        {progress && <pre style={{ whiteSpace: "pre-wrap", marginTop: 8 }}>{progress}</pre>}
        {running && (
          <p style={{ marginTop: 8 }}>
            Running case {liveIndex + 1}/{BAKEOFF_CASES.length}: {BAKEOFF_CASES[liveIndex]?.id}
          </p>
        )}
        {error && <pre style={{ whiteSpace: "pre-wrap", color: "red", marginTop: 8 }}>{error}</pre>}
      </section>

      {/* Per-case results for the current run */}
      {results.length > 0 && (
        <section style={{ marginTop: 24 }}>
          <h2 style={{ fontSize: 14, fontWeight: 700 }}>
            Results — {modelId} ({results.filter((r) => r.pass).length}/{results.length} passed)
          </h2>
          {results.map((r) => {
            const c = BAKEOFF_CASES.find((bc) => bc.id === r.caseId)!;
            return (
              <details key={r.caseId} style={{ marginTop: 8, border: "1px solid #333", padding: 8 }}>
                <summary style={{ color: r.pass ? "#0f0" : "#f55", cursor: "pointer" }}>
                  {r.pass ? "✓" : "✗"} {r.caseId} — {c.instruction} ({r.elapsedMs.toFixed(0)}ms
                  {r.reason ? `, ${r.reason}` : ""})
                </summary>
                {r.resolvedBlock && (
                  <div style={{ marginTop: 8, fontSize: 12 }}>
                    <p style={{ opacity: 0.6 }}>Resolved references injected into the prompt:</p>
                    <pre style={{ whiteSpace: "pre-wrap", background: "#1a1a1a", padding: 4 }}>{r.resolvedBlock}</pre>
                  </div>
                )}
                {!r.pass && (
                  <div style={{ marginTop: 8, fontSize: 12 }}>
                    <p style={{ opacity: 0.6 }}>Expected splits:</p>
                    <pre>{JSON.stringify(r.expectedSplits ?? "n/a", null, 2)}</pre>
                    <p style={{ opacity: 0.6 }}>Actual splits:</p>
                    <pre>{JSON.stringify(r.actualSplits ?? "n/a", null, 2)}</pre>
                    <p style={{ opacity: 0.6 }}>Raw model output:</p>
                    <pre style={{ whiteSpace: "pre-wrap" }}>{r.rawOutput ?? "n/a"}</pre>
                  </div>
                )}
              </details>
            );
          })}
        </section>
      )}
    </main>
  );
}
