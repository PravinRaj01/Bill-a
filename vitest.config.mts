import { defineConfig } from "vitest/config";
import path from "node:path";

export default defineConfig({
  test: {
    environment: "node",
    include: ["**/*.test.ts"],
    exclude: ["node_modules", ".next", "legacy"],
  },
  resolve: {
    alias: {
      // Mirror tsconfig.json's "@/*": ["./*"] so tests can import with the
      // same paths the app code uses.
      "@": path.resolve(import.meta.dirname, "."),
    },
  },
});
