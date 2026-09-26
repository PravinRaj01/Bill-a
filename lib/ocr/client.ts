import { createTesseractEngine } from "./engines/tesseract";
import { parseReceiptLines, type ParsedReceipt } from "./parse-lines";
import { prepareImage, type PreparedImage } from "./preprocess";
import type { OcrEngine, OcrOutput } from "./types";

// Browser entry point for local receipt scanning: photo in, parsed receipt out,
// no network. The worker, WASM core and language data are self-hosted under
// /ocr/ (copied from node_modules by scripts/copy-ocr-assets.mjs).
//
// Engine choice (tesseract.js + greyscale/contrast-stretched 1600 px input) comes
// from bench/results/ocr-bench.md: best or tied-best on every accuracy column and
// about twice as fast as the closest PaddleOCR model.

export type ScanStage = "preparing" | "loading-reader" | "reading";

export interface ScanResult {
  prepared: PreparedImage;
  ocr: OcrOutput;
  parsed: ParsedReceipt;
}

let engine: OcrEngine | null = null;
let ready: Promise<OcrEngine> | null = null;

/** Start loading the reader in the background (call when the scan screen opens). */
export function warmReceiptReader(): Promise<OcrEngine> {
  if (!ready) {
    const e = createTesseractEngine({
      workerPath: "/ocr/worker.min.js",
      corePath: "/ocr/core",
      langPath: "/ocr/lang",
    });
    ready = e.init().then(
      () => (engine = e),
      (err) => {
        ready = null; // let the next scan retry instead of caching the failure
        throw err;
      },
    );
  }
  return ready;
}

export async function releaseReceiptReader(): Promise<void> {
  const pending = ready;
  engine = null;
  ready = null;
  // If init is still in flight, wait for it so the worker it creates isn't leaked.
  const e = await pending?.catch(() => null);
  await e?.dispose();
}

export async function scanReceipt(file: Blob, onStage?: (s: ScanStage) => void): Promise<ScanResult> {
  onStage?.("preparing");
  const prepared = await prepareImage(file);

  onStage?.("loading-reader");
  const reader = await warmReceiptReader();

  onStage?.("reading");
  const ocr = await reader.recognize(await prepared.forOcr.arrayBuffer(), {
    width: prepared.width,
    height: prepared.height,
  });
  return { prepared, ocr, parsed: parseReceiptLines(ocr) };
}
