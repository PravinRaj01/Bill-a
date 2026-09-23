"use client";

import {
  CreateWebWorkerMLCEngine,
  hasModelInCache,
  prebuiltAppConfig,
  type AppConfig,
  type InitProgressReport,
  type MLCEngineInterface,
} from "@mlc-ai/web-llm";

// Some prebuilt model records ship a default config that WebLLM itself
// rejects at load time. gemma3-1b-it-q4f16_1-MLC is one: its record sets
// BOTH context_window_size (4096) and sliding_window_size (512) to
// positive values, and the engine requires exactly one of the two to be
// active — the other must be -1. This isn't something we're getting
// wrong; it's a bug in that model's prebuilt entry, worked around per
// the engine's own error message ("Consider modifying
// ModelRecord.overrides to set one of them to -1"). We keep the full
// 4096 context (our bake-off cases have longer item lists than the
// 512-token sliding window would comfortably hold) and disable the
// sliding window instead.
const MODEL_RECORD_FIXES: Record<string, { context_window_size?: number; sliding_window_size?: number }> = {
  "gemma3-1b-it-q4f16_1-MLC": { context_window_size: 4096, sliding_window_size: -1 },
};

// Default Cache API backend (not IndexedDB — that flag was removed from
// the library). See plan §7.3: this is what gives us "download once,
// instant + offline on every later visit."
export const appConfig: AppConfig = {
  ...prebuiltAppConfig,
  cacheBackend: "cache",
  model_list: prebuiltAppConfig.model_list.map((record) => {
    const fix = MODEL_RECORD_FIXES[record.model_id];
    if (!fix) return record;
    return { ...record, overrides: { ...record.overrides, ...fix } };
  }),
};

export interface EngineHandle {
  engine: MLCEngineInterface;
  /**
   * MUST be called before creating another engine in the same page —
   * otherwise the previous model's Worker keeps running with its GPU
   * buffers still allocated. This was a real bug, not a hypothesis: the
   * Phase 1 bake-off's second run showed later-loaded models in the same
   * session taking 20-30x longer (23-63s/case vs <2s) despite generating
   * a SHORT, unchanged-length output — i.e. it wasn't the model being
   * slow, it was GPU resource exhaustion from every previously loaded
   * model's Worker never being released. `engine.unload()` alone frees
   * the model weights from GPU memory; `worker.terminate()` also stops
   * the Worker thread itself so nothing lingers.
   */
  dispose: () => Promise<void>;
}

/**
 * Spins up the LLM in a dedicated Web Worker (never a Service Worker —
 * the browser can kill those without notice, which is unacceptable
 * mid-split). Must be called from a Client Component. Always call the
 * returned dispose() before creating a replacement engine — see
 * EngineHandle's doc comment.
 */
export async function createEngine(
  modelId: string,
  onProgress: (report: InitProgressReport) => void,
): Promise<EngineHandle> {
  const worker = new Worker(new URL("../../workers/llm.worker.ts", import.meta.url), {
    type: "module",
  });

  const engine = await CreateWebWorkerMLCEngine(worker, modelId, {
    appConfig,
    initProgressCallback: onProgress,
  });

  const dispose = async () => {
    try {
      await engine.unload();
    } catch {
      // Best-effort — the worker.terminate() below is what actually
      // guarantees the GPU resources and thread are released even if
      // unload() itself fails for some reason.
    }
    worker.terminate();
  };

  return { engine, dispose };
}

export const isModelCached = (modelId: string) => hasModelInCache(modelId, appConfig);
