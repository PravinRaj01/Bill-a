"use client";

// Phase 0 spike — not user-facing product UI.
//
// Validates the one thing the whole migration depends on: a Web Worker
// running WebLLM over WebGPU, producing schema-constrained JSON, surviving
// BOTH `next dev` and a production `next build && next start` under
// Turbopack. See plan §9 risk #1/#2 — there is no upstream-validated
// Next 16 + Turbopack + WebLLM example, so this is exploratory until it
// passes both checks.
//
// Uses Llama-3.2-1B-Instruct-q4f16_1-MLC because its download size
// (646 MiB) is the one figure already verified against the HF manifest;
// the other bake-off candidates (Phase 1) haven't been measured yet.

import { useState, useCallback } from "react";
import { createEngine, isModelCached } from "@/lib/ai/engine-client";
import { buildAssignmentPlanSchema, type AssignmentPlan } from "@/lib/ai/schemas";
import type { InitProgressReport, MLCEngineInterface } from "@mlc-ai/web-llm";

const MODEL_ID = "Llama-3.2-1B-Instruct-q4f16_1-MLC";

const SAMPLE_ITEMS = [
  { index: 0, name: "NASI LEMAK SPECIAL", quantity: 1 },
  { index: 1, name: "ICED LEMON TEA", quantity: 2 },
  { index: 2, name: "TEH TARIK", quantity: 1 },
  { index: 3, name: "CHICKEN RENDANG", quantity: 1 },
];
const SAMPLE_PEOPLE = ["Pravin", "Aisha", "Sarah"];
const SAMPLE_INSTRUCTION = "Pravin pays for the drinks, split the rest equally";

type Phase = "idle" | "checking-gpu" | "loading" | "ready" | "generating" | "error";

export default function WebLLMSpikePage() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [gpuInfo, setGpuInfo] = useState<string>("");
  const [progress, setProgress] = useState<string>("");
  const [wasCached, setWasCached] = useState<boolean | null>(null);
  const [engine, setEngine] = useState<MLCEngineInterface | null>(null);
  const [streamed, setStreamed] = useState("");
  const [parsed, setParsed] = useState<AssignmentPlan | null>(null);
  const [error, setError] = useState<string>("");
  const [tokPerSec, setTokPerSec] = useState<number | null>(null);
  const [sentSchema, setSentSchema] = useState<string>("");

  const checkGpu = useCallback(async () => {
    setPhase("checking-gpu");
    setError("");
    if (!("gpu" in navigator)) {
      setGpuInfo("navigator.gpu is undefined — no WebGPU in this browser (expected on iOS Safari; this is what forces tier D).");
      setPhase("idle");
      return;
    }
    try {
      // @ts-expect-error - navigator.gpu is not yet in all lib.dom.d.ts versions
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        setGpuInfo("navigator.gpu exists but requestAdapter() returned null — no usable GPU.");
        setPhase("idle");
        return;
      }
      const limits = adapter.limits;
      setGpuInfo(
        `WebGPU adapter OK. maxBufferSize=${(limits.maxBufferSize / 1e6).toFixed(0)}MB, ` +
          `maxStorageBufferBindingSize=${(limits.maxStorageBufferBindingSize / 1e6).toFixed(0)}MB`,
      );
    } catch (e) {
      setGpuInfo(`requestAdapter() threw: ${String(e)}`);
    }
    setPhase("idle");
  }, []);

  const loadModel = useCallback(async () => {
    setPhase("loading");
    setError("");
    try {
      const cached = await isModelCached(MODEL_ID);
      setWasCached(cached);

      const eng = await createEngine(MODEL_ID, (report: InitProgressReport) => {
        setProgress(report.text);
      });
      setEngine(eng);
      setPhase("ready");
    } catch (e) {
      setError(`Load failed: ${e instanceof Error ? e.stack ?? e.message : String(e)}`);
      setPhase("error");
    }
  }, []);

  const runSpike = useCallback(async () => {
    if (!engine) return;
    setPhase("generating");
    setError("");
    setStreamed("");
    setParsed(null);
    setTokPerSec(null);

    const candidateIndices = SAMPLE_ITEMS.map((i) => i.index);
    const schema = buildAssignmentPlanSchema(candidateIndices, SAMPLE_PEOPLE);
    // Diagnostic: prove exactly what's being sent, so a stale build/worker
    // chunk vs. an XGrammar keyword-support gap can be told apart at a
    // glance, without digging through DevTools' Sources panel.
    const schemaStr = JSON.stringify(schema);
    setSentSchema(schemaStr);
    console.log("[spike] schema sent to engine:", schemaStr);

    const menu = SAMPLE_ITEMS.map(
      (i) => `  [${i.index}] ${i.name} (qty ${i.quantity})`,
    ).join("\n");

    const messages = [
      {
        // WebLLM's JSON mode enforces the grammar during decoding, but the
        // docs are explicit: you must ALSO describe the schema in the
        // prompt yourself, or a model can spin generating whitespace until
        // it hits max_tokens. Restating the exact shape here is not
        // decorative — it's required. The grammar itself (schema, above)
        // is what actually bounds itemIndex/people to valid values; this
        // text is a second, redundant line of defense.
        role: "system" as const,
        content:
          "You assign receipt items to people. You never calculate money. " +
          "Respond with a single JSON object matching exactly:\n" +
          '{"assignments":[{"itemIndex":<int>,"people":["<name>",...],"weights":[<number>,...]}],' +
          '"defaultRule":"equal"|"exclude","notes":"<string>"}\n' +
          `itemIndex must be one of ${JSON.stringify(candidateIndices)}. ` +
          `people must be drawn only from ${JSON.stringify(SAMPLE_PEOPLE)}, no duplicates. ` +
          "Every person named in the instruction must appear in the people list; " +
          "ignore unknown names. Items not mentioned follow defaultRule. " +
          '"weights" is optional and parallel to "people"; omit it for an even share. ' +
          "Emit each item index at most once.",
      },
      {
        role: "user" as const,
        content: `PEOPLE: ${JSON.stringify(SAMPLE_PEOPLE)}\n\nCANDIDATE ITEMS (index, name, quantity):\n${menu}\n\nINSTRUCTION: ${JSON.stringify(SAMPLE_INSTRUCTION)}`,
      },
    ];

    try {
      const t0 = performance.now();
      const stream = await engine.chat.completions.create({
        messages,
        temperature: 0,
        stream: true,
        max_tokens: 512, // safety cap — see the system-message comment above
        response_format: {
          type: "json_object",
          schema: schemaStr,
        },
      });

      let buf = "";
      let tokenCount = 0;
      for await (const chunk of stream) {
        const delta = chunk.choices[0]?.delta?.content ?? "";
        buf += delta;
        tokenCount += 1;
        setStreamed(buf);
      }
      const elapsedSec = (performance.now() - t0) / 1000;
      setTokPerSec(tokenCount / elapsedSec);

      const plan = JSON.parse(buf) as AssignmentPlan;
      setParsed(plan);
      setPhase("ready");
    } catch (e) {
      setError(`Generation failed: ${e instanceof Error ? e.stack ?? e.message : String(e)}`);
      setPhase("error");
    }
  }, [engine]);

  return (
    <main style={{ maxWidth: 720, margin: "0 auto", padding: 24, fontFamily: "monospace", fontSize: 13 }}>
      <h1 style={{ fontSize: 18, fontWeight: 700 }}>WebLLM Spike (Phase 0 — dev only)</h1>
      <p style={{ opacity: 0.7 }}>
        Validates: Web Worker + WebGPU + schema-constrained JSON, under Turbopack, in dev and
        production builds. Not product UI.
      </p>

      <section style={{ marginTop: 16 }}>
        <button onClick={checkGpu} disabled={phase === "checking-gpu"}>
          1. Check WebGPU
        </button>
        {gpuInfo && <pre style={{ whiteSpace: "pre-wrap", marginTop: 8 }}>{gpuInfo}</pre>}
      </section>

      <section style={{ marginTop: 16 }}>
        <button onClick={loadModel} disabled={phase === "loading" || phase === "generating"}>
          2. Load {MODEL_ID}
        </button>
        {wasCached !== null && (
          <p style={{ marginTop: 8 }}>Was already cached: {String(wasCached)}</p>
        )}
        {progress && <pre style={{ whiteSpace: "pre-wrap", marginTop: 8 }}>{progress}</pre>}
      </section>

      <section style={{ marginTop: 16 }}>
        <button onClick={runSpike} disabled={!engine || phase === "generating"}>
          3. Run schema-constrained split ({SAMPLE_INSTRUCTION})
        </button>
        {sentSchema && (
          <>
            <p style={{ marginTop: 8, opacity: 0.6 }}>
              Schema sent this run (look for "enum" and "maxItems" — if absent, the browser is
              still running old code):
            </p>
            <pre style={{ whiteSpace: "pre-wrap", background: "#222", color: "#fc0", padding: 8, fontSize: 11 }}>
              {sentSchema}
            </pre>
          </>
        )}
        {streamed && (
          <>
            <p style={{ marginTop: 8, opacity: 0.6 }}>Raw stream:</p>
            <pre style={{ whiteSpace: "pre-wrap", background: "#111", color: "#0f0", padding: 8 }}>
              {streamed}
            </pre>
          </>
        )}
        {tokPerSec !== null && <p>~{tokPerSec.toFixed(1)} tok/s</p>}
        {parsed && (
          <>
            <p style={{ marginTop: 8, opacity: 0.6 }}>Parsed AssignmentPlan (JSON.parse succeeded):</p>
            <pre style={{ whiteSpace: "pre-wrap", background: "#eee", padding: 8 }}>
              {JSON.stringify(parsed, null, 2)}
            </pre>
          </>
        )}
      </section>

      {error && (
        <pre style={{ whiteSpace: "pre-wrap", color: "red", marginTop: 16 }}>{error}</pre>
      )}

      <p style={{ marginTop: 24, opacity: 0.5 }}>Phase: {phase}</p>
    </main>
  );
}
