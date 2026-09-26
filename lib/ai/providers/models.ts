// The ONLY place model ids live. Both providers retire and rename models every few
// months (Groq's llama-3.3-70b-versatile and Gemini 1.5/2.5 Flash are already gone
// or restricted for new users), so a rename is a one-line change here.
//
// Verified against the providers' docs in Sept 2026:
//  - Groq: openai/gpt-oss-120b and -20b are production models and support strict
//    structured outputs (`response_format: json_schema`, `strict: true`).
//  - Gemini: gemini-3.5-flash-lite is the stable low-cost tier with image input.
//    generateContent is still documented (not deprecated) alongside the newer
//    Interactions API.
//
// Measured live (bench/live-check.mts --bakeoff, 35 cases, Sept 2026):
//   gpt-oss-20b    35/35 correct, p50 576 ms, p95 778 ms
//   gemini flash-lite 34/35, p50 ~12 s under sustained load (3-12 s per call; free tier)
//   gpt-oss-120b   31/35 correct (+1 schema error), p50 794 ms, p95 1352 ms
// The smaller Groq model was both the fastest and the most accurate, and uses fewer
// tokens against the free tier's 8,000/min. The 120b is kept as a second attempt: Groq
// rate limits are per model, so a 429 on one may not apply to the other. (35 cases is
// a small sample — re-run the bake-off before changing these.)

export const MODELS = {
  groq: {
    primary: "openai/gpt-oss-20b",
    secondary: "openai/gpt-oss-120b",
  },
  gemini: {
    primary: "gemini-3.5-flash-lite",
  },
} as const;

export const ENDPOINTS = {
  groqChat: "https://api.groq.com/openai/v1/chat/completions",
  groqModels: "https://api.groq.com/openai/v1/models",
  geminiBase: "https://generativelanguage.googleapis.com/v1beta",
} as const;

/** Every origin a key is ever sent to. next.config.ts builds the CSP connect-src from this. */
export const PROVIDER_ORIGINS = ["https://api.groq.com", "https://generativelanguage.googleapis.com"] as const;
