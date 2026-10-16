import { afterEach, describe, expect, it, vi } from "vitest";

import { receiptFilename, shareImage } from "./share-image";

describe("receiptFilename", () => {
  it("makes a safe file name from the title", () => {
    expect(receiptFilename("Dinner out!")).toBe("dinner-out-receipt.png");
    expect(receiptFilename("  Café / Lunch  ")).toBe("caf-lunch-receipt.png");
    expect(receiptFilename("")).toBe("bill-receipt.png");
    expect(receiptFilename("日本")).toBe("bill-receipt.png");
    expect(receiptFilename("x".repeat(100)).length).toBeLessThanOrEqual(40 + "-receipt.png".length);
  });
});

describe("shareImage", () => {
  const blob = new Blob(["png"], { type: "image/png" });
  afterEach(() => vi.unstubAllGlobals());

  it("shares the file when the browser can", async () => {
    const share = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal("navigator", { canShare: () => true, share });
    expect(await shareImage(blob, "Dinner")).toBe("shared");
    const arg = share.mock.calls[0][0] as { files: File[]; title: string };
    expect(arg.files[0].name).toBe("dinner-receipt.png");
    expect(arg.files[0].type).toBe("image/png");
  });

  it("treats closing the share sheet as cancelled, not a download", async () => {
    const share = vi.fn().mockRejectedValue(new DOMException("closed", "AbortError"));
    vi.stubGlobal("navigator", { canShare: () => true, share });
    expect(await shareImage(blob, "Dinner")).toBe("cancelled");
  });
});

describe("isWebKit", () => {
  it("is true for Safari on iPhone and Mac, false for other engines", async () => {
    const { isWebKit } = await import("./share-image");
    expect(isWebKit("Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1")).toBe(true);
    expect(isWebKit("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15")).toBe(true);
    expect(isWebKit("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36")).toBe(false);
    // Chrome and Firefox on iPhone are WebKit underneath
    expect(isWebKit("Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/130.0 Mobile/15E148 Safari/604.1")).toBe(true);
    expect(isWebKit("Mozilla/5.0 (Android 14; Mobile) AppleWebKit/537.36 Chrome/130.0 Mobile Safari/537.36")).toBe(false);
  });
});
