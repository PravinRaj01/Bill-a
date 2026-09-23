// WebLLM worker entrypoint. Runs the model off the main thread so the UI
// stays responsive during load and generation.
//
// Phase 0 spike: this file only needs to exist and be reachable from a
// production `next build` — see app/dev/webllm-spike/page.tsx for the
// thing actually being validated (worker + WebGPU + schema-constrained
// JSON, surviving Turbopack's worker chunking).
import { WebWorkerMLCEngineHandler } from "@mlc-ai/web-llm";

const handler = new WebWorkerMLCEngineHandler();

self.onmessage = (msg: MessageEvent) => {
  handler.onmessage(msg);
};
