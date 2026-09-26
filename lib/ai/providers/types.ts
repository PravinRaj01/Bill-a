import type { ReceiptItem } from "@/types/domain";

export type ProviderId = "groq" | "gemini";

/** What a provider needs to turn an instruction into an AssignmentPlan. Never contains prices. */
export interface PlanRequest {
  people: string[];
  items: ReceiptItem[];
  /** One instruction, or a conversation applied in order (later overrides earlier). */
  instructions: string[];
  /** Output of resolveInstruction().promptBlock — already-worked-out item references. */
  resolvedBlock?: string;
}

export type ProviderErrorKind =
  | "auth" //          401/403: the key is wrong, revoked, or not allowed
  | "rate-limit" //    429
  | "server" //        5xx / 408
  | "model-gone" //    404: the model id was retired or renamed
  | "bad-request" //   400/422: the provider rejected our request or schema
  | "blocked" //       content policy / safety block
  | "bad-output" //    200 but empty, truncated, or not the JSON we asked for
  | "timeout"
  | "network"; //      offline, DNS, CORS, or a CSP block

export class ProviderError extends Error {
  constructor(
    public readonly provider: ProviderId,
    public readonly kind: ProviderErrorKind,
    message: string,
    public readonly status?: number,
    /** From the Retry-After header on a 429, when the provider sent one. */
    public readonly retryAfterMs?: number,
  ) {
    super(message);
    this.name = "ProviderError";
  }
}

export interface CallContext {
  signal?: AbortSignal;
  /** Injected for tests; defaults to the global fetch. */
  fetchImpl?: typeof fetch;
  /** Override the model id (the bake-off compares several). */
  model?: string;
}

/** Milliseconds from a Retry-After header (seconds), or undefined. Capped so a bad header can't stall the app. */
export function retryAfterMs(res: Response): number | undefined {
  const secs = Number(res.headers.get("retry-after"));
  return Number.isFinite(secs) && secs > 0 ? Math.min(secs, 120) * 1000 : undefined;
}

/** Maps an HTTP status to what the cascade should do about it. */
export function kindForStatus(status: number): ProviderErrorKind {
  if (status === 401 || status === 403) return "auth";
  if (status === 429) return "rate-limit";
  if (status === 404) return "model-gone";
  if (status === 408 || status >= 500) return "server";
  return "bad-request";
}

/** Wraps fetch so transport failures and aborts become ProviderErrors, never raw exceptions. */
export async function callFetch(
  provider: ProviderId,
  url: string,
  init: RequestInit,
  ctx: CallContext,
): Promise<Response> {
  const doFetch = ctx.fetchImpl ?? fetch;
  try {
    return await doFetch(url, { ...init, signal: ctx.signal });
  } catch (e) {
    if ((e as { name?: string })?.name === "AbortError") {
      throw new ProviderError(provider, "timeout", "The request timed out.");
    }
    // Also what a blocked CORS preflight or a CSP violation looks like from JS.
    throw new ProviderError(provider, "network", "Couldn't reach the service.");
  }
}
