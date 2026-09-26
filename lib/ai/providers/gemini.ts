import { buildAssignmentPlanSchema } from "../schemas";
import { buildItemMenu, buildSystemPrompt, buildUserPrompt } from "../prompts";
import { ENDPOINTS, MODELS } from "./models";
import { callFetch, kindForStatus, ProviderError, retryAfterMs, type CallContext, type PlanRequest } from "./types";

// Gemini, called DIRECTLY from the browser with the user's own key. The key goes
// in the `x-goog-api-key` header (never the URL, which ends up in logs and history).
//
// Uses generateContent with generationConfig.responseMimeType + responseSchema,
// which Google's reference still documents. Thinking is left at the model's default
// (minimal for the flash-lite tier): the thinking-control field is documented for
// the newer Interactions API but I couldn't confirm its generateContent spelling,
// and an unknown field would be a 400 on every call.

export function geminiUrl(model: string) {
  return `${ENDPOINTS.geminiBase}/models/${model}:generateContent`;
}

export function buildGeminiBody(req: PlanRequest) {
  const candidateIndices = req.items.map((_, i) => i);
  return {
    systemInstruction: { parts: [{ text: buildSystemPrompt(candidateIndices, req.people) }] },
    contents: [
      {
        role: "user",
        parts: [
          {
            text: buildUserPrompt(req.people, buildItemMenu(req.items), req.instructions, req.resolvedBlock),
          },
        ],
      },
    ],
    generationConfig: {
      temperature: 0,
      maxOutputTokens: 1024,
      responseMimeType: "application/json",
      responseSchema: buildAssignmentPlanSchema(candidateIndices, req.people, "gemini"),
    },
  };
}

interface GeminiResponse {
  candidates?: { content?: { parts?: { text?: string }[] }; finishReason?: string }[];
  promptFeedback?: { blockReason?: string };
}

/** Pulls the answer text out of a generateContent response, or throws the right ProviderError. */
export function readGeminiText(json: GeminiResponse): string {
  if (json.promptFeedback?.blockReason) {
    throw new ProviderError("gemini", "blocked", `Gemini blocked the request (${json.promptFeedback.blockReason}).`);
  }
  const cand = json.candidates?.[0];
  if (cand?.finishReason === "MAX_TOKENS") {
    throw new ProviderError("gemini", "bad-output", "Gemini's answer was cut off.");
  }
  if (cand?.finishReason && !["STOP", "FINISH_REASON_UNSPECIFIED"].includes(cand.finishReason)) {
    throw new ProviderError("gemini", "blocked", `Gemini stopped early (${cand.finishReason}).`);
  }
  const text = cand?.content?.parts?.map((p) => p.text ?? "").join("");
  if (!text) throw new ProviderError("gemini", "bad-output", "Gemini returned an empty answer.");
  return text;
}

/**
 * Gemini answers a bad key with HTTP 400 ("API key not valid"), not 401 — so a bare
 * status would tell the user "rejected the request" when the real problem is their key.
 * Read the error body to tell the two apart.
 */
export async function geminiFailure(res: Response): Promise<ProviderError> {
  let kind = kindForStatus(res.status);
  if (kind === "bad-request") {
    const body = await res.text().catch(() => "");
    if (/API key not valid|API_KEY_INVALID|API key expired/i.test(body)) kind = "auth";
  }
  return new ProviderError("gemini", kind, `Gemini responded ${res.status}`, res.status, retryAfterMs(res));
}

export async function geminiPlan(req: PlanRequest, apiKey: string, ctx: CallContext = {}): Promise<string> {
  const res = await callFetch(
    "gemini",
    geminiUrl(ctx.model ?? MODELS.gemini.primary),
    {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-goog-api-key": apiKey },
      body: JSON.stringify(buildGeminiBody(req)),
    },
    ctx,
  );
  if (!res.ok) throw await geminiFailure(res);
  let json: GeminiResponse;
  try {
    json = await res.json();
  } catch {
    throw new ProviderError("gemini", "bad-output", "Gemini returned something that wasn't JSON.");
  }
  return readGeminiText(json);
}
