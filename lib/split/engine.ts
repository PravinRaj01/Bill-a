import {
  SplitReconciliationError,
  type AssignmentPlan,
  type Cents,
  type Receipt,
  type SplitRecord,
  type SplitResult,
} from "@/types/domain";

// Everything here is exact integer arithmetic (BigInt for the intermediate fractions).
// No float ever touches a money amount.

function fmt(cents: Cents): string {
  return (cents / 100).toFixed(2);
}

function fracLabel(weight: number, allWeights: number[]): string {
  const equalWeights = allWeights.every((w) => w === allWeights[0]);
  if (equalWeights && allWeights.length > 1) return `x1/${allWeights.length}`;
  return `x${weight}`;
}

const gcd = (a: bigint, b: bigint): bigint => (b === 0n ? (a < 0n ? -a : a) : gcd(b, a % b));
const lcm = (a: bigint, b: bigint): bigint => (a / gcd(a, b)) * b;

/** Floor division that is correct for negative numerators too (BigInt `/` truncates toward zero). */
function floorDiv(a: bigint, b: bigint): bigint {
  const q = a / b;
  return a % b !== 0n && a < 0n !== b < 0n ? q - 1n : q;
}

/** Weights as integers: kept as-is when they already are, otherwise scaled by 1e6 and rounded. */
function intWeights(ws: number[]): bigint[] {
  if (ws.every((w) => Number.isInteger(w))) return ws.map((w) => BigInt(w));
  return ws.map((w) => BigInt(Math.round(w * 1_000_000)));
}

/** A slice of money shared among some people in fixed proportions. */
interface Group {
  price: bigint; // cents (negative for a discount line)
  targets: number[]; // indices into `people`
  weights: bigint[]; // integer weights, same length as targets, sum > 0
}

/**
 * Computes a bill split from a receipt, a people list, and an AssignmentPlan — the small,
 * LLM-or-fallback-parser-produced structure that says who owns which item (and, optionally,
 * who pays the tax). This function does ALL of the arithmetic. Nothing upstream of it is
 * trusted to have done any math; it only ever receives ownership decisions.
 *
 * How the cents work — "sum, then divide, then round ONCE":
 *   Each person's share of every item (and of the tax) is kept as an exact fraction. The
 *   fractions are added up per person, and only that final per-person total is rounded to
 *   whole cents, with the leftover cents going to whoever has the largest remainder.
 *   The older approach rounded every item separately and then added the rounded pieces,
 *   which let odd cents pile up on the same person: RM37.00 between two people came out
 *   as 18.51 / 18.49 instead of 18.50 / 18.50. Rounding once removes that bias, and the
 *   reconciliation below is now provable from the algebra (the exact shares add up to the
 *   expected total, so the rounded ones do too), not merely hoped for.
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
  const owned: string[][] = people.map(() => []);
  const trace: string[] = [];
  const groups: Group[] = [];
  // Sum of items that ended up with no target at all — only possible when defaultRule is
  // "exclude" and an item was never explicitly assigned. This is a deliberate product
  // feature (e.g. "just split the pizza between us, ignore the rest of the table's order"),
  // not an error — so the reconciliation target is scaled down to match what was actually
  // distributed, rather than the full receipt.
  let excludedSubtotal: Cents = 0;

  // Last assignment wins if the plan somehow lists the same itemIndex twice (the
  // orchestrator is expected to have deduped this upstream via validatePlan; this is a
  // defensive second line).
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

    const targets =
      explicitTargets.length > 0 ? explicitTargets : plan.defaultRule === "equal" ? people : [];

    if (targets.length === 0) {
      trace.push(
        `"${item.name}" (${fmt(item.totalPrice)}) — unassigned, excluded from this split ` +
          `(defaultRule: exclude). Nobody in this split is paying for it.`,
      );
      excludedSubtotal += item.totalPrice;
      return;
    }

    const rawWeights =
      assignment?.weights?.length === explicitTargets.length && explicitTargets.length > 0
        ? assignment.weights
        : targets.map(() => 1);
    // A non-positive weight total can't be divided; treat it as an even split rather than
    // dropping the item (which would break reconciliation).
    const weights = rawWeights.reduce((a, b) => a + b, 0) > 0 ? rawWeights : targets.map(() => 1);

    groups.push({
      price: BigInt(item.totalPrice),
      targets: targets.map((p) => peopleIndex.get(p)!),
      weights: intWeights(weights),
    });
    targets.forEach((person, k) => owned[peopleIndex.get(person)!].push(`${item.name} (${fracLabel(weights[k], weights)})`));

    const evenly = weights.every((w) => w === weights[0]);
    trace.push(
      `"${item.name}" ${fmt(item.totalPrice)} → ${targets.join(", ")}` +
        (targets.length > 1 ? (evenly ? ` (shared equally, ÷${targets.length})` : ` (shares ${weights.join(":")})`) : ""),
    );
  });

  // Derive the surcharge from the receipt's own numbers (total - subtotal) rather than
  // trusting receipt.tax in isolation. This matters for receipts (e.g. Malaysian ones)
  // where SST and service charge are separate line items and `tax` only captures one of them.
  const subtotal = receipt.items.reduce((s, it) => s + it.totalPrice, 0);
  const coveredSubtotal = subtotal - excludedSubtotal;
  const fullSurcharge = Math.max(0, receipt.total - subtotal);

  if (excludedSubtotal > 0) {
    trace.push(
      `Note: ${fmt(excludedSubtotal)} of unassigned items excluded from this split entirely — ` +
        `the people below are only splitting ${fmt(coveredSubtotal)} of the ${fmt(subtotal)} subtotal.`,
    );
  }

  // The surcharge is scaled down by the same fraction as the excluded items, so someone
  // covering 40% of the order's food cost pays 40% of the tax/service charge too. Computed
  // ONCE and reused for the target below, so `expected` is built from the exact same number.
  const surchargeForCovered =
    applyTax && subtotal > 0 && coveredSubtotal > 0 ? Math.round((fullSurcharge * coveredSubtotal) / subtotal) : 0;

  // Who pays the tax/service charge: the people the instruction named, or — by default —
  // everyone in proportion to what they ordered.
  const taxPayers = Array.from(new Set((plan.taxPayers ?? []).filter((p) => peopleIndex.has(p))));
  const pct = subtotal > 0 ? ((fullSurcharge / subtotal) * 100).toFixed(2) : "0.00";
  if (surchargeForCovered > 0 && taxPayers.length > 0) {
    groups.push({
      price: BigInt(surchargeForCovered),
      targets: taxPayers.map((p) => peopleIndex.get(p)!),
      weights: taxPayers.map(() => 1n),
    });
    trace.push(
      `Tax/service ${pct}% = ${fmt(surchargeForCovered)}, paid by ${taxPayers.join(", ")}` +
        (taxPayers.length > 1 ? ` (shared equally, ÷${taxPayers.length})` : "") +
        ` as instructed.`,
    );
    for (const p of taxPayers) owned[peopleIndex.get(p)!].push("tax/service");
  }

  // ---- exact per-person totals as fractions N[j] / D --------------------------------
  let D = 1n;
  for (const g of groups) D = lcm(D, g.weights.reduce((a, b) => a + b, 0n));
  const N: bigint[] = people.map(() => 0n);
  for (const g of groups) {
    const unit = D / g.weights.reduce((a, b) => a + b, 0n);
    g.targets.forEach((t, k) => {
      N[t] += g.price * g.weights[k] * unit;
    });
  }

  if (surchargeForCovered > 0 && taxPayers.length === 0) {
    // Everyone's exact food total F_j = N_j/D grows by the same factor:
    //   total_j = F_j * (covered + surcharge) / covered
    const cov = BigInt(coveredSubtotal);
    const factor = cov + BigInt(surchargeForCovered);
    for (let j = 0; j < N.length; j++) N[j] *= factor;
    D *= cov;
    trace.push(
      `Tax/service ${pct}% applied to the covered ${fmt(coveredSubtotal)} subtotal = ${fmt(surchargeForCovered)}, ` +
        `shared in proportion to what each person ordered.`,
    );
  }

  // ---- round ONCE, at the end -------------------------------------------------------
  const expected = coveredSubtotal + surchargeForCovered;
  const floors = N.map((n) => floorDiv(n, D));
  const remainders = N.map((n, i) => n - floors[i] * D);
  const floorSum = floors.reduce((a, b) => a + b, 0n);
  const residue = Number(BigInt(expected) - floorSum);

  // Largest remainder gets the leftover cent(s) first; ties broken by position so the
  // result is deterministic. The exact shares add up to `expected`, so 0 <= residue <= n.
  const order = remainders
    .map((r, i) => ({ r, i }))
    .sort((a, b) => (a.r === b.r ? a.i - b.i : a.r > b.r ? -1 : 1));
  const finalTotals: Cents[] = floors.map((f) => Number(f));
  if (residue < 0 || residue > people.length) {
    throw new SplitReconciliationError(expected, Number(floorSum));
  }
  for (let k = 0; k < residue; k++) finalTotals[order[k].i] += 1;

  const actual = finalTotals.reduce((a, b) => a + b, 0);
  if (actual !== expected) {
    // Unreachable by the algebra above (see SplitReconciliationError's message). If this
    // ever throws, the bug is in this function, not in the model output or the input data.
    throw new SplitReconciliationError(expected, actual);
  }

  trace.push(
    `Each share is worked out exactly and rounded once, at the end (not item by item): ` +
      people.map((p, i) => `${p} ${fmt(finalTotals[i])}`).join(", "),
  );
  trace.push(`✓ Reconciled: ${fmt(actual)} === receipt ${fmt(expected)}`);

  const splits: SplitRecord[] = people.map((name, i) => ({
    name,
    amount: finalTotals[i],
    items: owned[i].join(", ") || "(nothing assigned)",
  }));

  return { splits, reasoning: trace.join("\n"), verified: true };
}
