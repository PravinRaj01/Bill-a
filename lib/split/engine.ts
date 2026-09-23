import { apportion } from "./apportion";
import {
  SplitReconciliationError,
  type AssignmentPlan,
  type Cents,
  type Receipt,
  type SplitRecord,
  type SplitResult,
} from "@/types/domain";

function fmt(cents: Cents): string {
  return (cents / 100).toFixed(2);
}

function fracLabel(weight: number, allWeights: number[]): string {
  const equalWeights = allWeights.every((w) => w === allWeights[0]);
  if (equalWeights && allWeights.length > 1) return `x1/${allWeights.length}`;
  return `x${weight}`;
}

/**
 * Computes a bill split from a receipt, a people list, and an
 * AssignmentPlan — the small, LLM-or-fallback-parser-produced structure
 * that says who owns which item. This function does ALL of the
 * arithmetic. Nothing upstream of it is trusted to have done any math;
 * it only ever receives item ownership decisions.
 *
 * Every code path here uses apportion() (integer cents, Hamilton's
 * largest-remainder method), so the final assertion below is provably
 * unreachable rather than hopefully unreachable — see
 * SplitReconciliationError's message.
 */
export function computeSplit(
  receipt: Receipt,
  people: string[],
  plan: AssignmentPlan,
  applyTax: boolean,
): SplitResult {
  if (people.length === 0) {
    throw new Error("computeSplit: people list is empty");
  }

  const peopleIndex = new Map(people.map((p, i) => [p, i]));
  const rawTotals: Cents[] = people.map(() => 0);
  const owned: string[][] = people.map(() => []);
  const trace: string[] = [];
  // Sum of items that ended up with no target at all — only possible
  // when defaultRule is "exclude" and an item was never explicitly
  // assigned. This is a deliberate product feature (e.g. "just split the
  // pizza between us, ignore the rest of the table's order"), not an
  // error — so the reconciliation target below is scaled down to match
  // what was actually distributed, rather than the full receipt.
  let excludedSubtotal: Cents = 0;

  // Last assignment wins if the plan somehow lists the same itemIndex
  // twice (XGrammar's maxItems bounds the array length but can't enforce
  // uniqueItems — see plan §2.2 — so the orchestrator is expected to have
  // already deduped this upstream; this is a defensive second line).
  const byItemIndex = new Map<number, { people: string[]; weights?: number[] }>();
  for (const a of plan.assignments) {
    byItemIndex.set(a.itemIndex, { people: a.people, weights: a.weights });
  }

  receipt.items.forEach((item, i) => {
    const assignment = byItemIndex.get(i);

    // Dedupe + drop unknown names defensively (same rationale as above).
    const explicitTargets = assignment
      ? Array.from(new Set(assignment.people.filter((p) => peopleIndex.has(p))))
      : [];

    const targets = explicitTargets.length > 0
      ? explicitTargets
      : plan.defaultRule === "equal"
        ? people
        : [];

    if (targets.length === 0) {
      trace.push(
        `"${item.name}" (${fmt(item.totalPrice)}) — unassigned, excluded from this split ` +
          `(defaultRule: exclude). Nobody in this split is paying for it.`,
      );
      excludedSubtotal += item.totalPrice;
      return;
    }

    const weights =
      assignment?.weights?.length === explicitTargets.length && explicitTargets.length > 0
        ? assignment.weights
        : targets.map(() => 1);

    const shares = apportion(item.totalPrice, weights);

    targets.forEach((person, k) => {
      const idx = peopleIndex.get(person)!;
      rawTotals[idx] += shares[k];
      owned[idx].push(`${item.name} (${fracLabel(weights[k], weights)})`);
    });

    trace.push(
      `"${item.name}" ${fmt(item.totalPrice)} → ${targets.join(", ")} = ${shares.map(fmt).join(" / ")}`,
    );
  });

  // Derive the surcharge from the receipt's own numbers (total - subtotal)
  // rather than trusting receipt.tax in isolation. This matters for
  // receipts (e.g. Malaysian ones) where SST and service charge are
  // separate line items and `tax` only captures one of them — the exact
  // case the old prompt tried, and failed, to describe in prose.
  const subtotal = receipt.items.reduce((s, it) => s + it.totalPrice, 0);
  const coveredSubtotal = subtotal - excludedSubtotal;
  const fullSurcharge = Math.max(0, receipt.total - subtotal);

  if (excludedSubtotal > 0) {
    trace.push(
      `Note: ${fmt(excludedSubtotal)} of unassigned items excluded from this split entirely — ` +
        `the people below are only splitting ${fmt(coveredSubtotal)} of the ${fmt(subtotal)} subtotal.`,
    );
  }

  // The surcharge is scaled down by the same fraction as the excluded
  // items, so someone covering (say) 40% of the order's food cost pays
  // 40% of the tax/service charge too — not the tax on items they never
  // ordered. Computing this ONCE and reusing it for both the apportion()
  // call and the reconciliation target is what keeps the invariant below
  // provable rather than approximate: `expected` and `actual` are built
  // from the exact same number, not two independent derivations.
  const surchargeForCovered =
    applyTax && subtotal > 0 ? Math.round((fullSurcharge * coveredSubtotal) / subtotal) : 0;

  let finalTotals = rawTotals;
  if (surchargeForCovered > 0) {
    const prorated = apportion(surchargeForCovered, rawTotals);
    finalTotals = rawTotals.map((r, i) => r + prorated[i]);
    const pct = subtotal > 0 ? ((fullSurcharge / subtotal) * 100).toFixed(2) : "0.00";
    trace.push(
      `Tax/service ${pct}% applied to the covered ${fmt(coveredSubtotal)} subtotal = ` +
        `${fmt(surchargeForCovered)}, prorated by food cost: ${prorated.map(fmt).join(" / ")}`,
    );
  }

  const expected = coveredSubtotal + surchargeForCovered;
  const actual = finalTotals.reduce((a, b) => a + b, 0);

  if (actual !== expected) {
    // Unreachable given apportion()'s invariant. If this ever throws in
    // practice, the bug is in apportion() or the surcharge derivation
    // above, not in the model output or the input data — see the error's
    // own message.
    throw new SplitReconciliationError(expected, actual);
  }

  trace.push(`✓ Reconciled: ${fmt(actual)} === receipt ${fmt(expected)}`);

  const splits: SplitRecord[] = people.map((name, i) => ({
    name,
    amount: finalTotals[i],
    items: owned[i].join(", ") || "(nothing assigned)",
  }));

  return { splits, reasoning: trace.join("\n"), verified: true };
}
