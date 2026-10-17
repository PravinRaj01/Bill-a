import { afterEach, describe, expect, it, vi } from "vitest";

import { shareToPerson } from "./share";

afterEach(() => vi.unstubAllGlobals());

describe("shareToPerson", () => {
  it("opens the person's own WhatsApp chat when there is a number", async () => {
    const open = vi.fn(() => ({}));
    vi.stubGlobal("window", { open });
    vi.stubGlobal("navigator", { share: vi.fn() });
    expect(await shareToPerson("Hi! You owe RM10", "+60123456789")).toBe("opened");
    expect(open).toHaveBeenCalledWith("https://wa.me/60123456789?text=Hi!%20You%20owe%20RM10", "_blank", "noopener,noreferrer");
  });

  it("falls back to the share sheet without a number", async () => {
    const share = vi.fn(async () => {});
    const open = vi.fn();
    vi.stubGlobal("window", { open });
    vi.stubGlobal("navigator", { share });
    expect(await shareToPerson("Hi", null)).toBe("shared");
    expect(share).toHaveBeenCalledWith({ text: "Hi" });
    expect(open).not.toHaveBeenCalled();
  });

  it("falls back to the share sheet when the popup is blocked", async () => {
    const share = vi.fn(async () => {});
    vi.stubGlobal("window", { open: vi.fn(() => null) });
    vi.stubGlobal("navigator", { share });
    expect(await shareToPerson("Hi", "+60123456789")).toBe("shared");
  });
});
