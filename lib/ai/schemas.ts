// The only shape the LLM is ever allowed to produce. It assigns receipt
// items to people; it never computes a price. See lib/split/engine.ts
// (Phase 1) for where the actual arithmetic happens.
//
// The schema is built PER REQUEST, bounded to the actual candidate items
// and people list. This isn't cosmetic: WebLLM enforces the JSON schema
// via XGrammar during decoding, token by token. An unbounded schema
// (bare `array`, bare `integer`) legally permits the model to keep
// emitting array elements forever — observed in the Phase 0 spike as a
// runaway repetition loop (itemIndex climbing past the real item count,
// "people" filling with duplicate names) that ran until max_tokens cut
// it off mid-string and JSON.parse failed on the truncated output.
// Binding itemIndex/people to enums and every array to maxItems makes
// that failure structurally impossible: the grammar simply won't
// generate a token that would violate the bound, so the model is forced
// to close the object correctly once it hits the real limits.
// Note: maxItems bounds the array's total LENGTH to the real item count,
// which is what stops runaway growth — but plain JSON Schema has no
// clean way to also enforce "itemIndex is unique across array elements"
// (that needs draft 2019+ contains/minContains, which WebLLM's XGrammar
// support may not track). A model could in principle still emit the same
// itemIndex twice within the bounded length. Left as a Phase 1 item:
// either validate-and-reject in split-orchestrator.ts before handing the
// plan to computeSplit(), or determine XGrammar's actual draft support
// and tighten this further.
export function buildAssignmentPlanSchema(candidateIndices: number[], peopleNames: string[]) {
  return {
    type: "object",
    properties: {
      assignments: {
        type: "array",
        maxItems: candidateIndices.length,
        items: {
          type: "object",
          properties: {
            itemIndex: { type: "integer", enum: candidateIndices },
            people: {
              type: "array",
              items: { type: "string", enum: peopleNames },
              minItems: 1,
              maxItems: peopleNames.length,
              uniqueItems: true,
            },
            weights: {
              type: "array",
              items: { type: "number", minimum: 0 },
              maxItems: peopleNames.length,
            },
          },
          required: ["itemIndex", "people"],
        },
      },
      defaultRule: { type: "string", enum: ["equal", "exclude"] },
      notes: { type: "string" },
    },
    required: ["assignments", "defaultRule", "notes"],
  } as const;
}

export interface AssignmentPlan {
  assignments: Array<{ itemIndex: number; people: string[]; weights?: number[] }>;
  defaultRule: "equal" | "exclude";
  notes: string;
}
