import { ENDPOINTS } from "./models";
import { callFetch, kindForStatus, ProviderError, type CallContext, type ProviderId } from "./types";

export type KeyStatus = "valid" | "invalid" | "rate-limited" | "unreachable";

/**
 * Checks a key with the cheapest authenticated call each provider offers: listing
 * models. No tokens are generated, so it never counts against the free quota.
 */
export async function testKey(provider: ProviderId, apiKey: string, ctx: CallContext = {}): Promise<KeyStatus> {
  const [url, headers] =
    provider === "groq"
      ? [ENDPOINTS.groqModels, { Authorization: `Bearer ${apiKey}` }]
      : [`${ENDPOINTS.geminiBase}/models?pageSize=1`, { "x-goog-api-key": apiKey }];
  try {
    const res = await callFetch(provider, url, { method: "GET", headers }, ctx);
    if (res.ok) return "valid";
    const kind = kindForStatus(res.status);
    if (kind === "auth" || res.status === 400) return "invalid"; // Google answers a malformed key with 400
    if (kind === "rate-limit") return "rate-limited"; // the key itself is fine
    return "unreachable";
  } catch (e) {
    if (e instanceof ProviderError) return "unreachable";
    throw e;
  }
}
