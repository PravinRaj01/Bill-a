import { buildAssignmentPlanSchema } from "../schemas";
import { buildItemMenu, buildSystemPrompt, buildUserPrompt } from "../prompts";
import { ENDPOINTS, MODELS } from "./models";
import { callFetch, kindForStatus, ProviderError, retryAfterMs, type CallContext, type PlanRequest } from "./types";

// Groq, called DIRECTLY from the browser with the user's own key (Groq allows CORS).
// The key is sent only to api.groq.com, never to our servers.
//
// gpt-oss-120b with strict structured outputs: the response is guaranteed to match
// the schema's shape. reasoning_effort "low" keeps latency down for what is a
// small classification-style task; include_reasoning false drops the chain of
// thought from the response (it would only cost tokens to transfer).

export function buildGroqBody(req: PlanRequest, model: string = MODELS.groq.primary) {
  const candidateIndices = req.items.map((_, i) => i);
  return {
    model,
    temperature: 0,
    // Measured live: a plan call uses ~110-265 completion tokens (30-100 of them
    // reasoning). Groq counts the RESERVED max against the free tier's 8,000
    // tokens-per-minute budget, so 2048 here allowed only ~3 splits a minute; 768 is
    // ~3x the observed worst case and leaves room for ~8.
    max_completion_tokens: 768,
    reasoning_effort: "low",
    include_reasoning: false,
    messages: [
      { role: "system", content: buildSystemPrompt(candidateIndices, req.people) },
      {
        role: "user",
        content: buildUserPrompt(req.people, buildItemMenu(req.items), req.instructions, req.resolvedBlock),
      },
    ],
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "assignment_plan",
        strict: true,
        schema: buildAssignmentPlanSchema(candidateIndices, req.people, "groq"),
      },
    },
  };
}

/** Returns the model's raw JSON text; validatePlan() decides whether to trust it. */
export async function groqPlan(req: PlanRequest, apiKey: string, ctx: CallContext = {}): Promise<string> {
  const res = await callFetch(
    "groq",
    ENDPOINTS.groqChat,
    {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
      body: JSON.stringify(buildGroqBody(req, ctx.model)),
    },
    ctx,
  );

  if (!res.ok) {
    // Keep the provider's own explanation (never contains our key): "the request was
    // rejected" is undiagnosable, "response_format json_schema is not supported by
    // this model" is a one-line fix.
    const detail = await res
      .json()
      .then((j: { error?: { message?: string } }) => j?.error?.message?.slice(0, 300))
      .catch(() => undefined);
    throw new ProviderError("groq", kindForStatus(res.status), `Groq responded ${res.status}${detail ? `: ${detail}` : ""}`, res.status, retryAfterMs(res));
  }

  let json: { choices?: { message?: { content?: string | null }; finish_reason?: string }[] };
  try {
    json = await res.json();
  } catch {
    throw new ProviderError("groq", "bad-output", "Groq returned something that wasn't JSON.");
  }
  const choice = json.choices?.[0];
  if (choice?.finish_reason === "length") {
    throw new ProviderError("groq", "bad-output", "Groq's answer was cut off.");
  }
  const content = choice?.message?.content;
  if (!content) throw new ProviderError("groq", "bad-output", "Groq returned an empty answer.");
  return content;
}
