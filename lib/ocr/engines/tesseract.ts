import type { OcrBox, OcrEngine, OcrLine, OcrOutput, OcrWord } from "../types";

// tesseract.js v7 (WASM). Word-level boxes come from the `blocks` output, which
// is off by default in v6+.

export interface TesseractOptions {
  /** ISO 639-3 code(s), e.g. "eng" or "eng+msa". */
  lang?: string;
  /** Self-hosted asset locations (required in the browser so nothing loads from a CDN). */
  workerPath?: string;
  corePath?: string;
  langPath?: string;
  /** Tesseract page-segmentation mode; "3" (auto) is the library default. */
  psm?: string;
}

interface TessWord { text: string; confidence: number; bbox: { x0: number; y0: number; x1: number; y1: number } }
interface TessLine { text: string; confidence: number; bbox: TessWord["bbox"]; words: TessWord[] }

const box = (b: TessWord["bbox"]): OcrBox => ({ x: b.x0, y: b.y0, w: b.x1 - b.x0, h: b.y1 - b.y0 });

export function createTesseractEngine(opts: TesseractOptions = {}): OcrEngine {
  let worker: import("tesseract.js").Worker | null = null;
  const name = `tesseract${opts.psm ? `-psm${opts.psm}` : ""}`;

  return {
    name,
    async init() {
      const { createWorker } = await import("tesseract.js");
      worker = await createWorker(opts.lang ?? "eng", 1, {
        ...(opts.workerPath ? { workerPath: opts.workerPath } : {}),
        ...(opts.corePath ? { corePath: opts.corePath } : {}),
        ...(opts.langPath ? { langPath: opts.langPath } : {}),
      });
      if (opts.psm) await worker.setParameters({ tessedit_pageseg_mode: opts.psm as never });
    },

    async recognize(image, size): Promise<OcrOutput> {
      if (!worker) throw new Error("tesseract engine not initialised");
      const t0 = performance.now();
      const res = await worker.recognize(image as unknown as Buffer, {}, { text: true, blocks: true });
      const ms = performance.now() - t0;

      const lines: OcrLine[] = [];
      for (const block of res.data.blocks ?? []) {
        for (const para of block.paragraphs ?? []) {
          for (const l of (para.lines ?? []) as unknown as TessLine[]) {
            const words: OcrWord[] = (l.words ?? [])
              .filter((w) => w.text.trim().length > 0)
              .map((w) => ({ text: w.text, box: box(w.bbox), confidence: w.confidence / 100 }));
            const text = l.text.replace(/\s+/g, " ").trim();
            if (!text) continue;
            lines.push({
              text,
              box: box(l.bbox),
              words,
              confidence: words.length ? words.reduce((s, w) => s + w.confidence, 0) / words.length : l.confidence / 100,
            });
          }
        }
      }
      lines.sort((a, b) => a.box.y - b.box.y);
      return { text: res.data.text, lines, width: size.width, height: size.height, ms };
    },

    async dispose() {
      await worker?.terminate();
      worker = null;
    },
  };
}
