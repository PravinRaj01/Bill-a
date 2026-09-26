import type { NextConfig } from "next";
import { PROVIDER_ORIGINS } from "./lib/ai/providers/models";

// Content-Security-Policy.
//
// Why it matters here: users paste their own Groq/Gemini keys and we keep them in
// localStorage, which any script on the page can read. The CSP's `connect-src` is
// what stops a script that somehow got injected from SENDING a key anywhere but the
// two provider APIs. Honest limits: script-src still needs 'unsafe-inline' (Next's
// inline bootstrap scripts; removing it means nonces, which force every page to
// render dynamically), so this reduces the XSS blast radius rather than making XSS
// impossible — the actual defence is not having injection bugs (React escapes by
// default and the app renders no user HTML).
//
// Each allowance, and why:
//  - 'wasm-unsafe-eval'        tesseract.js compiles a WebAssembly module (the OCR core)
//  - worker-src 'self' blob:   tesseract's worker, served from /ocr/
//  - connect-src PROVIDERS     direct browser -> Groq / Gemini calls with the user's key
//  - img-src googleusercontent Google profile pictures on the Account page
//  - form-action accounts.google.com
//                              Chrome applies form-action to the redirect that follows a
//                              form POST, and "Continue with Google" is a POST that 302s there
//  - dev only: 'unsafe-eval' and websockets for React's dev tooling / hot reload

const isDev = process.env.NODE_ENV !== "production";

const csp = [
  "default-src 'self'",
  `script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval'${isDev ? " 'unsafe-eval'" : ""}`,
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob: https://*.googleusercontent.com",
  "font-src 'self' data:",
  `connect-src 'self' ${PROVIDER_ORIGINS.join(" ")}${isDev ? " ws: wss:" : ""}`,
  "worker-src 'self' blob:",
  "manifest-src 'self'",
  "frame-src 'none'",
  "object-src 'none'",
  "base-uri 'self'",
  "form-action 'self' https://accounts.google.com",
  "frame-ancestors 'none'",
].join("; ");

const nextConfig: NextConfig = {
  // Native module (prebuilt per platform): keep it out of the server bundle so the Linux
  // binary that `npm ci` installs on Vercel is the one that gets loaded.
  serverExternalPackages: ["@node-rs/argon2"],
  async headers() {
    return [
      // The service worker must always be revalidated, or a bad version could stick.
      { source: "/sw.js", headers: [{ key: "Cache-Control", value: "no-cache, no-store, must-revalidate" }] },
      {
        source: "/:path*",
        headers: [
          { key: "Content-Security-Policy", value: csp },
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          { key: "Permissions-Policy", value: "geolocation=(), microphone=(), payment=()" },
        ],
      },
    ];
  },
};

export default nextConfig;
