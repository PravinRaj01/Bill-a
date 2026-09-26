// The JSON schema for types/domain.ts's AssignmentPlan — the only thing an LLM is
// ever allowed to produce. It assigns receipt items to people; it never computes a
// price (lib/split/engine.ts does every cent).
//
// Built PER REQUEST and bounded to the real items and people, so the model can't
// even express an out-of-range index or an unknown name. Constrained decoding
// guarantees shape, NOT meaning (JSON Schema can't say "itemIndex is unique across
// the array", and Gemini's docs say to always validate values) — so every response
// still goes through lib/ai/validatePlan.ts.
//
// Deliberately NOT in the schema: a `weights` field. The Phase 1 bake-off showed
// small models hallucinating weights that mirror the receipt `quantity`. The engine
// still supports weighted shares; custom ratios will be an explicit UI control, not
// something inferred from free text.

export type SchemaTarget = "groq" | "gemini";

/**
 * - groq: strict structured outputs — every object needs `additionalProperties:
 *   false` and every property in `required`; `enum` and `minItems`/`maxItems` are
 *   accepted (checked live). `uniqueItems` is NOT, despite the docs.
 * - gemini: `responseSchema` is an OpenAPI-style subset. `enum` is only reliable
 *   for strings, so itemIndex is a bounded integer (minimum/maximum) rather than an
 *   integer enum, and keywords Gemini's schema has rejected historically
 *   (`additionalProperties`, `uniqueItems`) are left out.
 */
export function buildAssignmentPlanSchema(
  candidateIndices: number[],
  peopleNames: string[],
  target: SchemaTarget = "groq",
) {
  if (candidateIndices.length === 0) throw new Error("buildAssignmentPlanSchema: no candidate items");
  if (peopleNames.length === 0) throw new Error("buildAssignmentPlanSchema: no people");

  const itemIndex =
    target === "groq"
      ? { type: "integer", enum: candidateIndices }
      : { type: "integer", minimum: Math.min(...candidateIndices), maximum: Math.max(...candidateIndices) };

  const people = {
    type: "array",
    items: { type: "string", enum: peopleNames },
    minItems: 1,
    maxItems: peopleNames.length,
    // No uniqueItems for either provider: Groq's strict mode rejects it outright
    // (HTTP 400 "uniqueItems is not supported" — found by the live check, and contrary
    // to the docs), and Gemini's schema subset doesn't take it either. validatePlan
    // dedupes names instead.
  };

  const closed = target === "groq" ? { additionalProperties: false } : {};

  return {
    type: "object",
    ...closed,
    properties: {
      assignments: {
        type: "array",
        maxItems: candidateIndices.length,
        items: {
          type: "object",
          ...closed,
          properties: { itemIndex, people },
          required: ["itemIndex", "people"],
        },
      },
      defaultRule: { type: "string", enum: ["equal", "exclude"] },
      // Who pays the tax / service charge. Empty = the default (everyone, in proportion to
      // what they ordered). Required (possibly empty) because Groq's strict mode needs every
      // property listed in `required`.
      taxPayers: {
        type: "array",
        items: { type: "string", enum: peopleNames },
        minItems: 0,
        maxItems: peopleNames.length,
      },
      notes: { type: "string" },
    },
    required: ["assignments", "defaultRule", "taxPayers", "notes"],
  } as const;
}

// Re-exported so call sites can import the type from one place; the canonical
// definition lives in types/domain.ts next to Receipt/SplitResult.
export type { AssignmentPlan } from "@/types/domain";
