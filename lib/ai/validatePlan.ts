import { z } from "zod";
import type { AssignmentPlan } from "@/types/domain";

// The gate every AssignmentPlan passes through before it reaches computeSplit(),
// whichever tier produced it (Groq, Gemini, or the fallback parser).
//
// Schema-constrained decoding guarantees SHAPE, not meaning: JSON Schema can't
// say "itemIndex is unique across the array", Gemini's docs tell you to always
// validate values yourself, and a model can still name someone who isn't in the
// bill. Cleanable slips are repaired here; anything that means the model misread
// the task is REJECTED so the cascade moves to the next tier instead of quietly
// computing a different split from the one the user asked for.

export type PlanValidation =
  | { ok: true; plan: AssignmentPlan; repairs: string[] }
  | { ok: false; reason: string };

const planSchema = z.object({
  assignments: z.array(
    z.object({
      itemIndex: z.number().int(),
      people: z.array(z.string()),
      weights: z.array(z.number().positive().finite()).optional(),
    }),
  ),
  defaultRule: z.enum(["equal", "exclude"]),
  notes: z.string().optional().default(""),
});

const norm = (s: string) => s.trim().toLowerCase();

export function validatePlan(
  raw: unknown,
  ctx: { itemCount: number; people: string[] },
): PlanValidation {
  let value = raw;
  if (typeof raw === "string") {
    try {
      value = JSON.parse(raw);
    } catch {
      return { ok: false, reason: "not valid JSON" };
    }
  }

  const parsed = planSchema.safeParse(value);
  if (!parsed.success) {
    return { ok: false, reason: `wrong shape: ${parsed.error.issues[0]?.message ?? "invalid"}` };
  }

  const repairs: string[] = [];
  // Canonical spelling comes from the bill's own people list.
  const canonical = new Map(ctx.people.map((p) => [norm(p), p]));
  const byIndex = new Map<number, { itemIndex: number; people: string[]; weights?: number[] }>();

  for (const a of parsed.data.assignments) {
    if (a.itemIndex < 0 || a.itemIndex >= ctx.itemCount) {
      return { ok: false, reason: `itemIndex ${a.itemIndex} is outside 0..${ctx.itemCount - 1}` };
    }

    const seen = new Set<string>();
    const people: string[] = [];
    let unknown = 0;
    for (const name of a.people) {
      const known = canonical.get(norm(name));
      if (!known) {
        unknown++;
        continue;
      }
      if (seen.has(known)) {
        repairs.push(`dropped duplicate "${known}" on item ${a.itemIndex}`);
        continue;
      }
      seen.add(known);
      people.push(known);
    }

    // A name we don't know is dropped — but if that leaves nobody, the model's
    // answer was about someone else entirely. Falling back to defaultRule for
    // that item would silently change who pays, so reject.
    if (people.length === 0) {
      return { ok: false, reason: `item ${a.itemIndex} is assigned to nobody in this bill` };
    }
    if (unknown > 0) repairs.push(`dropped ${unknown} unknown name(s) on item ${a.itemIndex}`);

    // Weights only make sense parallel to the surviving names; if names were
    // dropped or deduped they no longer line up, so drop them (even split).
    const weights =
      a.weights && a.weights.length === a.people.length && people.length === a.people.length
        ? a.weights
        : undefined;
    if (a.weights && !weights) repairs.push(`dropped misaligned weights on item ${a.itemIndex}`);

    if (byIndex.has(a.itemIndex)) repairs.push(`item ${a.itemIndex} assigned twice; last one wins`);
    byIndex.set(a.itemIndex, { itemIndex: a.itemIndex, people, ...(weights ? { weights } : {}) });
  }

  return {
    ok: true,
    plan: {
      assignments: [...byIndex.values()],
      defaultRule: parsed.data.defaultRule,
      notes: parsed.data.notes,
    },
    repairs,
  };
}
