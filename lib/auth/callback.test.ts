import { describe, expect, it } from "vitest";

import { safeCallback } from "./callback";

describe("safeCallback", () => {
  it("keeps a path inside the dashboard, with its query", () => {
    expect(safeCallback("/dashboard/add/alex")).toBe("/dashboard/add/alex");
    expect(safeCallback("/dashboard/new?how=scan")).toBe("/dashboard/new?how=scan");
    expect(safeCallback("/dashboard")).toBe("/dashboard");
  });
  it("reduces an absolute URL to its path, dropping the host", () => {
    expect(safeCallback("http://localhost:3100/dashboard/people")).toBe("/dashboard/people");
    expect(safeCallback("https://evil.example/dashboard/add/x")).toBe("/dashboard/add/x");
  });
  it("falls back for anything outside the dashboard or malformed", () => {
    for (const bad of ["https://evil.example/", "//evil.example/dashboard", "/", "/dashboardx", "/dashboard//evil", "javascript:alert(1)", "/dashboard\..", "", undefined, 5, "x".repeat(600)]) {
      expect(safeCallback(bad)).toBe("/dashboard");
    }
  });
});
