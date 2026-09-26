import type { OcrBox, OcrEngine, OcrLine, OcrOutput, OcrWord } from "../types";

// PaddleOCR (PP-OCRv6, ONNX) via ppu-paddle-ocr. The service class is injected so
// the same adapter runs on onnxruntime-node (benchmark) and onnxruntime-web
// (browser worker) — the two ship as different entry points of the package.

// ppu-paddle-ocr reports boxes as {x, y, width, height} (not w/h).
interface PaddleBox { x: number; y: number; width: number; height: number }
interface PaddleSegment { text: string; box: PaddleBox; confidence: number }
const toBox = (b: PaddleBox): OcrBox => ({ x: b.x, y: b.y, w: b.width, h: b.height });
interface PaddleResult { text: string; lines: PaddleSegment[][] }
interface PaddleService {
  initialize(): Promise<void>;
  recognize(image: ArrayBuffer, options?: { noCache?: boolean }): Promise<PaddleResult>;
  destroy(): Promise<void>;
}

export function createPaddleEngine(
  make: () => PaddleService,
  label = "paddle",
): OcrEngine {
  let service: PaddleService | null = null;

  return {
    name: label,
    async init() {
      service = make();
      await service.initialize();
    },

    async recognize(image, size): Promise<OcrOutput> {
      if (!service) throw new Error("paddle engine not initialised");
      const t0 = performance.now();
      // noCache: the library keeps a process-wide result cache keyed by image bytes,
      // which would hand one model's output to another and make timings meaningless.
      const res = await service.recognize(image, { noCache: true });
      const ms = performance.now() - t0;

      const lines: OcrLine[] = res.lines
        .filter((segs) => segs.length > 0)
        .map((segs) => {
          const ordered = [...segs].sort((a, b) => a.box.x - b.box.x);
          const words: OcrWord[] = ordered.map((s) => ({ text: s.text, box: toBox(s.box), confidence: s.confidence }));
          const x0 = Math.min(...words.map((w) => w.box.x));
          const y0 = Math.min(...words.map((w) => w.box.y));
          const x1 = Math.max(...words.map((w) => w.box.x + w.box.w));
          const y1 = Math.max(...words.map((w) => w.box.y + w.box.h));
          return {
            text: ordered.map((s) => s.text).join(" ").replace(/\s+/g, " ").trim(),
            box: { x: x0, y: y0, w: x1 - x0, h: y1 - y0 },
            words,
            confidence: words.reduce((s, w) => s + w.confidence, 0) / words.length,
          };
        })
        .sort((a, b) => a.box.y - b.box.y);

      return { text: res.text, lines, width: size.width, height: size.height, ms };
    },

    async dispose() {
      await service?.destroy();
      service = null;
    },
  };
}
