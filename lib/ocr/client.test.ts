import { afterEach, describe, expect, it, vi } from "vitest";

// A fake OCR engine, so the reader's lifecycle can be tested without WASM.
const h = vi.hoisted(() => {
  const created: {
    disposed: boolean;
    init: () => Promise<void>;
    recognize: () => Promise<unknown>;
    dispose: () => Promise<void>;
  }[] = [];
  const gate = { hold: null as null | Promise<void> };
  return { created, gate };
});

vi.mock("./engines/tesseract", () => ({
  createTesseractEngine: () => {
    const e = {
      name: "fake",
      disposed: false,
      init: async () => {},
      recognize: async () => {
        if (h.gate.hold) await h.gate.hold; // lets a test hold a scan "in flight"
        if (e.disposed) throw new Error("tesseract engine not initialised"); // what the real adapter does
        return { text: "TOTAL 1.00", lines: [], width: 10, height: 10, ms: 1 };
      },
      dispose: async () => {
        e.disposed = true;
      },
    };
    h.created.push(e);
    return e;
  },
}));

vi.mock("./preprocess", () => ({
  prepareImage: async () => ({ display: new Blob(["d"]), forOcr: new Blob(["o"]), width: 10, height: 10, originalBytes: 1 }),
}));

import { pinReceiptReader, releaseReceiptReader, releaseReceiptReaderIfIdle, scanReceipt, warmReceiptReader } from "./client";

afterEach(async () => {
  h.gate.hold = null;
  await releaseReceiptReader();
  h.created.length = 0;
});

describe("reader lifecycle (regression: offline priming disposed the reader mid-scan)", () => {
  it("frees the reader when nothing is using it", async () => {
    await warmReceiptReader();
    await releaseReceiptReaderIfIdle();
    expect(h.created[0].disposed).toBe(true);
  });

  it("does NOT free it while the scan screen has it pinned, and does once unpinned", async () => {
    const unpin = pinReceiptReader();
    await warmReceiptReader();
    await releaseReceiptReaderIfIdle();
    expect(h.created[0].disposed).toBe(false);
    unpin();
    await releaseReceiptReaderIfIdle();
    expect(h.created[0].disposed).toBe(true);
  });

  it("unpinning twice can't drive the count negative and free someone else's reader", async () => {
    const a = pinReceiptReader();
    const b = pinReceiptReader();
    a();
    a(); // second call is a no-op
    await warmReceiptReader();
    await releaseReceiptReaderIfIdle(); // b still holds it
    expect(h.created[0].disposed).toBe(false);
    b();
  });

  it("a scan in flight protects the reader from a concurrent priming release", async () => {
    let open!: () => void;
    h.gate.hold = new Promise<void>((r) => (open = r));

    const scan = scanReceipt(new Blob(["photo"], { type: "image/jpeg" }));
    await vi.waitFor(() => expect(h.created.length).toBe(1)); // scan has started warming
    await releaseReceiptReaderIfIdle(); // priming finishes right now
    expect(h.created[0].disposed).toBe(false);

    open();
    await expect(scan).resolves.toMatchObject({ ocr: { text: "TOTAL 1.00" } }); // no "not initialised"
    await releaseReceiptReaderIfIdle();
    expect(h.created[0].disposed).toBe(true); // and once the scan is done it IS freed
  });

  it("a scan that fails still unpins (the reader isn't leaked forever)", async () => {
    h.gate.hold = Promise.reject(new Error("boom"));
    h.gate.hold.catch(() => {});
    await expect(scanReceipt(new Blob(["x"]))).rejects.toThrow("boom");
    h.gate.hold = null;
    await warmReceiptReader();
    await releaseReceiptReaderIfIdle();
    expect(h.created.at(-1)!.disposed).toBe(true);
  });
});
