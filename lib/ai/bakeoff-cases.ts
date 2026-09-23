import type { AssignmentPlan, Receipt } from "@/types/domain";

/**
 * Phase 1 model bake-off — plan §3.2.
 *
 * Each case is a (receipt, people, instruction) triple plus a
 * hand-authored "correct" AssignmentPlan — the ground-truth
 * interpretation a careful human would produce. We never hand-compute
 * expected cent amounts directly (too easy to get wrong by hand); instead
 * the harness runs BOTH the expectedPlan and the model's own plan through
 * the same, already-tested computeSplit() and compares the resulting
 * per-person amounts. That isolates "did the model understand the
 * instruction" from "was the arithmetic right" — the arithmetic is
 * proven separately by lib/split/engine.test.ts.
 *
 * These deliberately give the model the FULL raw item list with no
 * retrieval pre-filtering (the embeddings/lexicon layer in plan §5.1
 * doesn't exist until Phase 4) — so this measures closer to a
 * worst-case signal than the real pipeline's eventual conditions. If a
 * model does well here, it will likely do at least as well once
 * retrieval hands it a pre-filtered candidate list.
 */

export interface BakeoffCase {
  id: string;
  /** What a user would actually type. */
  instruction: string;
  people: string[];
  receipt: Receipt;
  applyTax: boolean;
  /** Hand-authored ground-truth interpretation. */
  expectedPlan: AssignmentPlan;
}

const RM = (amount: number) => Math.round(amount * 100); // RM x.xx -> cents

export const BAKEOFF_CASES: BakeoffCase[] = [
  {
    id: "drinks-payer",
    instruction: "Pravin pays for the drinks, split the rest equally",
    people: ["Pravin", "Aisha", "Sarah"],
    receipt: {
      currency: "RM",
      tax: RM(1.7),
      total: RM(18.7),
      items: [
        { name: "NASI LEMAK SPECIAL", quantity: 1, unitPrice: RM(10), totalPrice: RM(10) },
        { name: "ICED LEMON TEA", quantity: 2, unitPrice: RM(3.5), totalPrice: RM(7) },
        { name: "TEH TARIK", quantity: 1, unitPrice: RM(4), totalPrice: RM(4) },
        { name: "CHICKEN RENDANG", quantity: 1, unitPrice: RM(12), totalPrice: RM(12) },
      ],
    },
    applyTax: true,
    expectedPlan: {
      assignments: [
        { itemIndex: 1, people: ["Pravin"] },
        { itemIndex: 2, people: ["Pravin"] },
      ],
      defaultRule: "equal",
      notes: "",
    },
  },
  {
    id: "equal-baseline",
    instruction: "Split everything equally",
    people: ["Pravin", "Aisha"],
    receipt: {
      currency: "RM",
      tax: 0,
      total: RM(20),
      items: [
        { name: "FRIED RICE", quantity: 1, unitPrice: RM(8), totalPrice: RM(8) },
        { name: "SPRING ROLLS", quantity: 1, unitPrice: RM(6), totalPrice: RM(6) },
        { name: "ICED MILO", quantity: 1, unitPrice: RM(6), totalPrice: RM(6) },
      ],
    },
    applyTax: false,
    expectedPlan: { assignments: [], defaultRule: "equal", notes: "" },
  },
  {
    id: "one-person-pays-everything",
    instruction: "Pravin is treating everyone today, put it all on him",
    people: ["Pravin", "Aisha", "Sarah"],
    receipt: {
      currency: "RM",
      tax: RM(2),
      total: RM(32),
      items: [
        { name: "MEE GORENG", quantity: 1, unitPrice: RM(9), totalPrice: RM(9) },
        { name: "ROTI CANAI", quantity: 2, unitPrice: RM(4), totalPrice: RM(8) },
        { name: "TEH TARIK", quantity: 3, unitPrice: RM(4.33), totalPrice: RM(13) },
      ],
    },
    applyTax: true,
    expectedPlan: {
      assignments: [
        { itemIndex: 0, people: ["Pravin"] },
        { itemIndex: 1, people: ["Pravin"] },
        { itemIndex: 2, people: ["Pravin"] },
      ],
      defaultRule: "equal",
      notes: "",
    },
  },
  {
    id: "two-way-item-split",
    instruction: "Split the pizza between Pravin and Aisha only, everyone splits the rest equally",
    people: ["Pravin", "Aisha", "Sarah"],
    receipt: {
      currency: "RM",
      tax: 0,
      total: RM(45),
      items: [
        { name: "PIZZA MARGHERITA", quantity: 1, unitPrice: RM(25), totalPrice: RM(25) },
        { name: "CAESAR SALAD", quantity: 1, unitPrice: RM(12), totalPrice: RM(12) },
        { name: "GARLIC BREAD", quantity: 1, unitPrice: RM(8), totalPrice: RM(8) },
      ],
    },
    applyTax: false,
    expectedPlan: {
      assignments: [{ itemIndex: 0, people: ["Pravin", "Aisha"] }],
      defaultRule: "equal",
      notes: "",
    },
  },
  {
    id: "category-desserts",
    instruction: "Sarah pays for all the desserts, split the rest equally",
    people: ["Pravin", "Sarah"],
    receipt: {
      currency: "RM",
      tax: 0,
      total: RM(38),
      items: [
        { name: "BURGER DELUXE", quantity: 1, unitPrice: RM(18), totalPrice: RM(18) },
        { name: "FRENCH FRIES", quantity: 1, unitPrice: RM(8), totalPrice: RM(8) },
        { name: "CHOCOLATE LAVA CAKE", quantity: 1, unitPrice: RM(9), totalPrice: RM(9) },
        { name: "LEMON SORBET", quantity: 1, unitPrice: RM(3), totalPrice: RM(3) },
      ],
    },
    applyTax: false,
    expectedPlan: {
      assignments: [
        { itemIndex: 2, people: ["Sarah"] },
        { itemIndex: 3, people: ["Sarah"] },
      ],
      defaultRule: "equal",
      notes: "",
    },
  },
  {
    id: "all-items-explicitly-named",
    instruction: "Pravin had the rice, Aisha had the chicken, Sarah had the tea — everyone covers their own",
    people: ["Pravin", "Aisha", "Sarah"],
    receipt: {
      currency: "RM",
      tax: 0,
      total: RM(30),
      items: [
        { name: "FRIED RICE", quantity: 1, unitPrice: RM(9), totalPrice: RM(9) },
        { name: "CHICKEN RENDANG", quantity: 1, unitPrice: RM(13), totalPrice: RM(13) },
        { name: "TEH TARIK", quantity: 2, unitPrice: RM(4), totalPrice: RM(8) },
      ],
    },
    applyTax: false,
    expectedPlan: {
      assignments: [
        { itemIndex: 0, people: ["Pravin"] },
        { itemIndex: 1, people: ["Aisha"] },
        { itemIndex: 2, people: ["Sarah"] },
      ],
      defaultRule: "exclude",
      notes: "",
    },
  },
  {
    // Replaces an earlier "weighted-by-quantity" case that asked for an
    // explicit ratio (e.g. 2:1) — that capability was removed from the
    // schema after two rounds of models hallucinating weights that
    // mirror the item's receipt quantity rather than a real ratio (see
    // lib/ai/schemas.ts and lib/ai/prompts.ts's revision-history
    // comments). This tests a different, still-in-scope pattern instead:
    // one person excluded from a single item while defaultRule "equal"
    // still covers everything else — distinct from negation-one-item
    // (which excludes someone from an item nobody else is excluded
    // from) because here the excluded person still shares OTHER items.
    id: "single-item-exclusion",
    instruction: "Everyone splits equally, but Sarah's vegetarian so she's not paying for the chicken",
    people: ["Pravin", "Aisha", "Sarah"],
    receipt: {
      currency: "RM",
      tax: 0,
      total: RM(30),
      items: [
        { name: "FRIED RICE", quantity: 1, unitPrice: RM(9), totalPrice: RM(9) },
        { name: "CHICKEN RENDANG", quantity: 1, unitPrice: RM(12), totalPrice: RM(12) },
        { name: "VEG CURRY", quantity: 1, unitPrice: RM(9), totalPrice: RM(9) },
      ],
    },
    applyTax: false,
    expectedPlan: {
      assignments: [{ itemIndex: 1, people: ["Pravin", "Aisha"] }],
      defaultRule: "equal",
      notes: "",
    },
  },
  {
    id: "exclude-unmentioned-items",
    instruction: "Just split the burger between Pravin and Sarah, forget about the rest of the table's order",
    people: ["Pravin", "Sarah"],
    receipt: {
      currency: "RM",
      tax: 0,
      total: RM(40),
      items: [
        { name: "BURGER DELUXE", quantity: 1, unitPrice: RM(18), totalPrice: RM(18) },
        { name: "FISH AND CHIPS", quantity: 1, unitPrice: RM(16), totalPrice: RM(16) },
        { name: "MILKSHAKE", quantity: 1, unitPrice: RM(6), totalPrice: RM(6) },
      ],
    },
    applyTax: false,
    expectedPlan: {
      assignments: [{ itemIndex: 0, people: ["Pravin", "Sarah"] }],
      defaultRule: "exclude",
      notes: "",
    },
  },
  {
    id: "single-item-receipt",
    instruction: "Split it equally between the two of us",
    people: ["Pravin", "Aisha"],
    receipt: {
      currency: "RM",
      tax: 0,
      total: RM(15),
      items: [{ name: "LARGE PIZZA", quantity: 1, unitPrice: RM(15), totalPrice: RM(15) }],
    },
    applyTax: false,
    expectedPlan: { assignments: [], defaultRule: "equal", notes: "" },
  },
  {
    id: "large-group-equal",
    instruction: "Everyone splits this equally, five of us",
    people: ["Pravin", "Aisha", "Sarah", "Wei", "Farah"],
    receipt: {
      currency: "RM",
      tax: RM(8),
      total: RM(88),
      items: [
        { name: "SET A", quantity: 1, unitPrice: RM(20), totalPrice: RM(20) },
        { name: "SET B", quantity: 1, unitPrice: RM(20), totalPrice: RM(20) },
        { name: "SET C", quantity: 1, unitPrice: RM(20), totalPrice: RM(20) },
        { name: "SET D", quantity: 1, unitPrice: RM(20), totalPrice: RM(20) },
      ],
    },
    applyTax: true,
    expectedPlan: { assignments: [], defaultRule: "equal", notes: "" },
  },
  {
    id: "informal-item-reference",
    instruction: "Pravin had the teh tarik, split the rest between everyone",
    people: ["Pravin", "Aisha", "Sarah"],
    receipt: {
      currency: "RM",
      tax: 0,
      total: RM(27),
      items: [
        { name: "NASI LEMAK SPECIAL", quantity: 1, unitPrice: RM(10), totalPrice: RM(10) },
        { name: "ICED LEMON TEA", quantity: 1, unitPrice: RM(4), totalPrice: RM(4) },
        { name: "TEH TARIK", quantity: 1, unitPrice: RM(4), totalPrice: RM(4) },
        { name: "CHICKEN RENDANG", quantity: 1, unitPrice: RM(9), totalPrice: RM(9) },
      ],
    },
    applyTax: false,
    expectedPlan: {
      assignments: [{ itemIndex: 2, people: ["Pravin"] }],
      defaultRule: "equal",
      notes: "",
    },
  },
  {
    id: "two-explicit-payers-plus-shared-dessert",
    instruction: "Pravin pays for the rice, Aisha pays for the chicken, and we all split the cake",
    people: ["Pravin", "Aisha", "Sarah"],
    receipt: {
      currency: "RM",
      tax: 0,
      total: RM(35),
      items: [
        { name: "FRIED RICE", quantity: 1, unitPrice: RM(9), totalPrice: RM(9) },
        { name: "CHICKEN RENDANG", quantity: 1, unitPrice: RM(13), totalPrice: RM(13) },
        { name: "CHOCOLATE LAVA CAKE", quantity: 1, unitPrice: RM(13), totalPrice: RM(13) },
      ],
    },
    applyTax: false,
    expectedPlan: {
      assignments: [
        { itemIndex: 0, people: ["Pravin"] },
        { itemIndex: 1, people: ["Aisha"] },
      ],
      defaultRule: "equal",
      notes: "",
    },
  },
  {
    id: "negation-one-item",
    instruction: "Everyone splits equally except the milkshake, which is just for Sarah",
    people: ["Pravin", "Aisha", "Sarah"],
    receipt: {
      currency: "RM",
      tax: 0,
      total: RM(24),
      items: [
        { name: "FRIES", quantity: 1, unitPrice: RM(8), totalPrice: RM(8) },
        { name: "NUGGETS", quantity: 1, unitPrice: RM(10), totalPrice: RM(10) },
        { name: "MILKSHAKE", quantity: 1, unitPrice: RM(6), totalPrice: RM(6) },
      ],
    },
    applyTax: false,
    expectedPlan: {
      assignments: [{ itemIndex: 2, people: ["Sarah"] }],
      defaultRule: "equal",
      notes: "",
    },
  },
  {
    id: "three-way-explicit-with-tax",
    instruction: "Split three ways, but Pravin also owes for his extra teh tarik",
    people: ["Pravin", "Aisha", "Sarah"],
    receipt: {
      currency: "RM",
      tax: RM(3),
      total: RM(38),
      items: [
        { name: "SET MEAL", quantity: 3, unitPrice: RM(10), totalPrice: RM(30) },
        { name: "TEH TARIK", quantity: 1, unitPrice: RM(5), totalPrice: RM(5) },
      ],
    },
    applyTax: true,
    expectedPlan: {
      assignments: [{ itemIndex: 1, people: ["Pravin"] }],
      defaultRule: "equal",
      notes: "",
    },
  },
  {
    id: "no-instruction-defaults-equal",
    instruction: "Split equally",
    people: ["Pravin", "Aisha", "Sarah"],
    receipt: {
      currency: "RM",
      tax: RM(1.5),
      total: RM(31.5),
      items: [
        { name: "MAIN COURSE", quantity: 3, unitPrice: RM(10), totalPrice: RM(30) },
      ],
    },
    applyTax: true,
    expectedPlan: { assignments: [], defaultRule: "equal", notes: "" },
  },
];
