// Client-side image preparation, run the moment the user picks a file.
//
// Why: the old flow base64-uploaded a full-resolution phone photo (several MB) to
// a cold-started server. Now the photo is shrunk once, in the browser, to a size
// that both the OCR engines and Gemini handle well, and nothing else is uploaded.
//
//   original photo ──► EXIF-rotate ──► longest edge ≤ 1600 px ──► JPEG q0.8   (display, Cloud Enhance)
//                                                        └──► greyscale + contrast stretch  (local OCR, if it helps)

export const MAX_EDGE = 1600;
export const JPEG_QUALITY = 0.8;

/** Scale (w, h) down so the longest edge is at most `max`; never scales up. */
export function fitWithin(w: number, h: number, max = MAX_EDGE): { width: number; height: number } {
  const longest = Math.max(w, h);
  if (longest <= max) return { width: w, height: h };
  const k = max / longest;
  return { width: Math.max(1, Math.round(w * k)), height: Math.max(1, Math.round(h * k)) };
}

/** Photos whose long edge is below this are "low resolution": the text is only a few pixels tall. */
export const LOW_RES_EDGE = 700;
/** ...and are enlarged to this for the OCR engine (tesseract wants text lines ~20+ px tall). */
export const ENLARGE_TO = 1400;

/**
 * Size to feed the OCR engine. Normal photos are used as-is (see fitWithin); a tiny one
 * is enlarged, because measured on a 338x450 receipt, tesseract read almost nothing at
 * native size and most of it at 3x. Enlarging EVERYTHING was tried on the benchmark and was
 * a wash (raw text improved, exact totals dipped 12 -> 11 of 18), so only tiny photos are touched.
 */
export function ocrSizeFor(w: number, h: number): { width: number; height: number; enlarged: boolean } {
  const longest = Math.max(w, h);
  if (longest >= LOW_RES_EDGE) return { ...fitWithin(w, h), enlarged: false };
  const k = ENLARGE_TO / longest;
  return { width: Math.round(w * k), height: Math.round(h * k), enlarged: true };
}

/**
 * In-place greyscale + percentile contrast stretch on RGBA pixels: luminance is
 * remapped so the darkest 1% maps to 0 and the brightest 1% to 255. Faded thermal
 * paper and dim photos gain the contrast the OCR needs; already-crisp images are
 * barely changed.
 */
export function greyscaleStretch(rgba: Uint8ClampedArray, clipFraction = 0.01): void {
  const n = rgba.length / 4;
  const hist = new Uint32Array(256);
  const luma = new Uint8Array(n);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const y = (rgba[p] * 299 + rgba[p + 1] * 587 + rgba[p + 2] * 114) / 1000;
    const v = y < 0 ? 0 : y > 255 ? 255 : Math.round(y);
    luma[i] = v;
    hist[v]++;
  }
  const clip = Math.floor(n * clipFraction);
  let lo = 0;
  for (let acc = 0; lo < 255 && acc + hist[lo] <= clip; lo++) acc += hist[lo];
  let hi = 255;
  for (let acc = 0; hi > 0 && acc + hist[hi] <= clip; hi--) acc += hist[hi];
  const span = Math.max(1, hi - lo);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const v = Math.round(((luma[i] - lo) * 255) / span);
    const c = v < 0 ? 0 : v > 255 ? 255 : v;
    rgba[p] = rgba[p + 1] = rgba[p + 2] = c;
    rgba[p + 3] = 255;
  }
}

export interface PreparedImage {
  /** Colour JPEG, ≤1600 px: for display and for Cloud Enhance. */
  display: Blob;
  /** Greyscale + stretched JPEG for the local OCR engine. */
  forOcr: Blob;
  width: number;
  height: number;
  /** Pixel size of `forOcr` (differs from width/height only when a tiny photo was enlarged). */
  ocrWidth: number;
  ocrHeight: number;
  /** The photo was too small to read reliably, however well we enlarge it. */
  lowRes: boolean;
  /** The original photo's size in pixels, for the "this photo is small" message. */
  sourceWidth: number;
  sourceHeight: number;
  /** Size of the file the user picked, for the "we shrank this" UI. */
  originalBytes: number;
}

type AnyCanvas = OffscreenCanvas | HTMLCanvasElement;

function makeCanvas(width: number, height: number): AnyCanvas {
  if (typeof OffscreenCanvas !== "undefined") return new OffscreenCanvas(width, height);
  const c = document.createElement("canvas");
  c.width = width;
  c.height = height;
  return c;
}

function toJpeg(canvas: AnyCanvas): Promise<Blob> {
  if ("convertToBlob" in canvas) return canvas.convertToBlob({ type: "image/jpeg", quality: JPEG_QUALITY });
  return new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("JPEG encode failed"))), "image/jpeg", JPEG_QUALITY),
  );
}

/**
 * Decode, honour EXIF orientation, downscale and produce both variants. Throws a
 * readable Error if the file isn't a decodable image (HEIC on browsers without
 * support, a PDF, a truncated download).
 */
export async function prepareImage(file: Blob): Promise<PreparedImage> {
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  } catch {
    throw new Error("Couldn't read that image. Try a JPEG or PNG photo.");
  }
  try {
    const { width, height } = fitWithin(bitmap.width, bitmap.height);
    const canvas = makeCanvas(width, height);
    const ctx = canvas.getContext("2d", { willReadFrequently: true }) as
      | CanvasRenderingContext2D
      | OffscreenCanvasRenderingContext2D
      | null;
    if (!ctx) throw new Error("This browser can't process images.");
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(bitmap, 0, 0, width, height);

    const display = await toJpeg(canvas);

    // The OCR copy: greyscale + contrast stretch, and enlarged first if the photo is tiny.
    const ocr = ocrSizeFor(bitmap.width, bitmap.height);
    let ocrCanvas = canvas;
    let ocrCtx = ctx;
    if (ocr.enlarged) {
      ocrCanvas = makeCanvas(ocr.width, ocr.height);
      ocrCtx = ocrCanvas.getContext("2d", { willReadFrequently: true }) as typeof ctx;
      if (!ocrCtx) throw new Error("This browser can't process images.");
      ocrCtx.imageSmoothingQuality = "high";
      ocrCtx.drawImage(bitmap, 0, 0, ocr.width, ocr.height);
    }
    const pixels = ocrCtx.getImageData(0, 0, ocr.width, ocr.height);
    greyscaleStretch(pixels.data);
    ocrCtx.putImageData(pixels, 0, 0);
    const forOcr = await toJpeg(ocrCanvas);

    return {
      display,
      forOcr,
      width,
      height,
      ocrWidth: ocr.width,
      ocrHeight: ocr.height,
      lowRes: ocr.enlarged,
      sourceWidth: bitmap.width,
      sourceHeight: bitmap.height,
      originalBytes: file.size,
    };
  } finally {
    bitmap.close();
  }
}
