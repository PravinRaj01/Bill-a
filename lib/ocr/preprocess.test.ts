import { describe, expect, it } from "vitest";
import { fitWithin, greyscaleStretch } from "./preprocess";

describe("fitWithin", () => {
  it.each([
    [4000, 3000, 1600, 1200],
    [3000, 4000, 1200, 1600],
    [1600, 900, 1600, 900],
    [800, 600, 800, 600], // never scales up
    [1601, 1, 1600, 1], // never collapses a dimension to 0
  ])("%ix%i -> %ix%i", (w, h, ew, eh) => {
    expect(fitWithin(w, h)).toEqual({ width: ew, height: eh });
  });
});

describe("greyscaleStretch", () => {
  const px = (v: number[]) => new Uint8ClampedArray(v.flatMap((y) => [y, y, y, 255]));

  it("stretches a low-contrast image to the full range", () => {
    const data = px([100, 100, 110, 120, 130, 140, 140, 140]); // faded: 100..140
    greyscaleStretch(data, 0);
    const ys = Array.from({ length: 8 }, (_, i) => data[i * 4]);
    expect(Math.min(...ys)).toBe(0);
    expect(Math.max(...ys)).toBe(255);
  });

  it("keeps ordering (a monotonic remap) and writes grey RGB with opaque alpha", () => {
    const data = px([10, 60, 120, 200]);
    greyscaleStretch(data, 0);
    for (let i = 0; i < 4; i++) {
      expect(data[i * 4]).toBe(data[i * 4 + 1]);
      expect(data[i * 4]).toBe(data[i * 4 + 2]);
      expect(data[i * 4 + 3]).toBe(255);
    }
    expect(data[0]).toBeLessThan(data[4]);
    expect(data[4]).toBeLessThan(data[8]);
    expect(data[8]).toBeLessThan(data[12]);
  });

  it("converts colour by luminance, not by averaging", () => {
    const data = new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255]);
    greyscaleStretch(data, 0);
    // green is perceptually brightest of the three primaries, blue darkest
    expect(data[4]).toBeGreaterThan(data[0]);
    expect(data[0]).toBeGreaterThan(data[8]);
  });

  it("handles a flat image without dividing by zero", () => {
    const data = px([128, 128, 128, 128]);
    expect(() => greyscaleStretch(data)).not.toThrow();
    for (const v of data) expect(Number.isFinite(v)).toBe(true);
  });

  it("clips outliers so one bright speck doesn't flatten the rest", () => {
    const ys = Array.from({ length: 200 }, (_, i) => (i === 0 ? 255 : 100 + (i % 20)));
    const data = px(ys);
    greyscaleStretch(data, 0.02);
    const spread = new Set(Array.from({ length: 200 }, (_, i) => data[i * 4])).size;
    expect(spread).toBeGreaterThan(10);
  });
});
