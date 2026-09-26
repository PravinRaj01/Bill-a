// Engine-neutral OCR types. Every engine adapter (tesseract.js, PaddleOCR) maps
// its own output into these, so parse-lines.ts, the benchmark and the UI never
// depend on which engine ran.

/** Pixel coordinates in the image the engine was given (origin top-left). */
export interface OcrBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** One recognised unit: a word (tesseract) or a text segment (PaddleOCR). */
export interface OcrWord {
  text: string;
  box: OcrBox;
  /** 0..1 */
  confidence: number;
}

export interface OcrLine {
  text: string;
  box: OcrBox;
  words: OcrWord[];
  /** 0..1, mean of the words (unweighted). */
  confidence: number;
}

export interface OcrOutput {
  text: string;
  lines: OcrLine[];
  /** Dimensions of the image that was recognised. */
  width: number;
  height: number;
  /** Wall time of recognize() alone, excluding init. */
  ms: number;
}

/**
 * Engines take ENCODED image bytes (JPEG/PNG) so the same adapter runs in Node
 * (benchmark), a Web Worker (app) and tests. Decoding is the engine's job; the
 * caller passes the pixel size because not every engine reports it.
 */
export interface OcrEngine {
  readonly name: string;
  init(): Promise<void>;
  recognize(image: ArrayBuffer, size: { width: number; height: number }): Promise<OcrOutput>;
  dispose(): Promise<void>;
}
