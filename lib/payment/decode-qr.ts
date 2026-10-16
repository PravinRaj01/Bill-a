// Reads the text out of a QR code in an image the user picked (a screenshot from their bank app). Done on
// the device: the image is never uploaded or stored, only the decoded text is kept. The browser's native
// BarcodeDetector is used where it exists; jsQR is the fallback (iOS Safari, Firefox).

async function toBitmap(file: Blob): Promise<ImageBitmap> {
  return createImageBitmap(file);
}

async function viaDetector(bitmap: ImageBitmap): Promise<string | null> {
  const Detector = (globalThis as unknown as { BarcodeDetector?: new (o: { formats: string[] }) => { detect: (i: ImageBitmap) => Promise<{ rawValue: string }[]> } }).BarcodeDetector;
  if (!Detector) return null;
  try {
    const codes = await new Detector({ formats: ["qr_code"] }).detect(bitmap);
    return codes[0]?.rawValue ?? null;
  } catch {
    return null;
  }
}

async function viaJsQr(bitmap: ImageBitmap): Promise<string | null> {
  // A bank-app screenshot is large; scaling it down keeps the scan quick and still readable.
  const MAX = 1400;
  const scale = Math.min(1, MAX / Math.max(bitmap.width, bitmap.height));
  const w = Math.max(1, Math.round(bitmap.width * scale));
  const h = Math.max(1, Math.round(bitmap.height * scale));
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return null;
  ctx.drawImage(bitmap, 0, 0, w, h);
  const { default: jsQR } = await import("jsqr");
  const { data } = ctx.getImageData(0, 0, w, h);
  return jsQR(data, w, h, { inversionAttempts: "attemptBoth" })?.data ?? null;
}

/** The QR's text, or null when the image has no readable QR code. Throws only if the file isn't an image. */
export async function decodeQrImage(file: Blob): Promise<string | null> {
  const bitmap = await toBitmap(file);
  try {
    return (await viaDetector(bitmap)) ?? (await viaJsQr(bitmap));
  } finally {
    bitmap.close();
  }
}
