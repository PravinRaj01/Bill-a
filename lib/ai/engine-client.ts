"use client";

import {
  CreateWebWorkerMLCEngine,
  hasModelInCache,
  prebuiltAppConfig,
  type AppConfig,
  type InitProgressReport,
  type MLCEngineInterface,
} from "@mlc-ai/web-llm";

// Default Cache API backend (not IndexedDB — that flag was removed from
// the library). See plan §7.3: this is what gives us "download once,
// instant + offline on every later visit."
export const appConfig: AppConfig = {
  ...prebuiltAppConfig,
  cacheBackend: "cache",
};

/**
 * Spins up the LLM in a dedicated Web Worker (never a Service Worker —
 * the browser can kill those without notice, which is unacceptable
 * mid-split). Must be called from a Client Component.
 */
export async function createEngine(
  modelId: string,
  onProgress: (report: InitProgressReport) => void,
): Promise<MLCEngineInterface> {
  const worker = new Worker(new URL("../../workers/llm.worker.ts", import.meta.url), {
    type: "module",
  });

  return CreateWebWorkerMLCEngine(worker, modelId, {
    appConfig,
    initProgressCallback: onProgress,
  });
}

export const isModelCached = (modelId: string) => hasModelInCache(modelId, appConfig);
