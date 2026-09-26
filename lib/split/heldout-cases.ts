import type { BakeoffCase } from "@/lib/ai/bakeoff-cases";
import type { Receipt } from "@/types/domain";

// Held-out evaluation set for the deterministic fallback parser.
//
// Written BEFORE lib/split/fallback-parser.ts existed, from a receipt and
// phrasings the parser was not designed against. The 15 bake-off cases are the
// parser's development set (it was built to pass them, so that score is
// optimistic by construction); this file is the honest generalisation number.
// Do not tune the parser to these — fix general bugs, then report both scores.
//
// `saferAsChip`: for phrasings a rule-based parser genuinely can't resolve
// (typos, unknown names, synonyms not in the lexicon), asking the user a
// question is an acceptable outcome — only a confidently WRONG plan is a fail.

const RM = (amount: number) => Math.round(amount * 100);

const mamak: Receipt = {
  currency: "RM",
  tax: 0,
  total: RM(42.5),
  items: [
    { name: "NASI GORENG", quantity: 1, unitPrice: RM(9), totalPrice: RM(9) }, // 0
    { name: "MEE GORENG", quantity: 1, unitPrice: RM(8), totalPrice: RM(8) }, // 1
    { name: "ROTI CANAI", quantity: 2, unitPrice: RM(4), totalPrice: RM(8) }, // 2
    { name: "TEH TARIK", quantity: 2, unitPrice: RM(4), totalPrice: RM(8) }, // 3
    { name: "MILO AIS", quantity: 1, unitPrice: RM(4.5), totalPrice: RM(4.5) }, // 4
    { name: "CENDOL", quantity: 1, unitPrice: RM(5), totalPrice: RM(5) }, // 5
  ],
};

const people = ["Pravin", "Aisha", "Wei", "Farah"];

export interface HeldOutCase extends BakeoffCase {
  saferAsChip?: boolean;
  /** The right answer IS a question (e.g. a person who isn't in the group). */
  mustAsk?: boolean;
}

const c = (
  id: string,
  instruction: string,
  expectedPlan: BakeoffCase["expectedPlan"],
  saferAsChip = false,
): HeldOutCase => ({ id, instruction, people, receipt: mamak, applyTax: false, expectedPlan, saferAsChip });

const eq = "equal" as const;
const ex = "exclude" as const;

// A receipt with tax/service (RM37.00 = 34.91 + 2.09 GST): the tax-payer cases need one.
const realfood: Receipt = {
  currency: "RM",
  tax: RM(2.09),
  total: RM(37),
  items: [
    { name: "MUSH NOODLES DRY", quantity: 1, unitPrice: RM(18.87), totalPrice: RM(18.87) }, // 0
    { name: "STEAM DUMPLINGS", quantity: 1, unitPrice: RM(16.04), totalPrice: RM(16.04) }, // 1
  ],
};
const couple = ["Pravin", "Wifey"];
const ct = (id: string, instruction: string, expectedPlan: BakeoffCase["expectedPlan"]): HeldOutCase => ({
  id, instruction, people: couple, receipt: realfood, applyTax: true, expectedPlan,
});

export const HELDOUT_CASES: HeldOutCase[] = [
  c("single-person-single-item", "Aisha had the nasi goreng", {
    assignments: [{ itemIndex: 0, people: ["Aisha"] }], defaultRule: eq, notes: "",
  }),
  c("category-drinks-one-payer", "Wei pays for all the drinks", {
    assignments: [{ itemIndex: 3, people: ["Wei"] }, { itemIndex: 4, people: ["Wei"] }], defaultRule: eq, notes: "",
  }),
  c("two-people-share-item-else-equal", "Farah and Aisha share the roti canai, everyone else splits the rest", {
    assignments: [{ itemIndex: 2, people: ["Farah", "Aisha"] }], defaultRule: eq, notes: "",
  }),
  c("negation-she-he-not-having", "Wei didn't have the cendol", {
    assignments: [{ itemIndex: 5, people: ["Pravin", "Aisha", "Farah"] }], defaultRule: eq, notes: "",
  }),
  c("treat-everyone", "Pravin treats everyone", {
    assignments: [0, 1, 2, 3, 4, 5].map((itemIndex) => ({ itemIndex, people: ["Pravin"] })), defaultRule: eq, notes: "",
  }),
  c("just-two-items-ignore-rest", "Just the mee goreng and the milo ais, split between Pravin and Wei, ignore everything else", {
    assignments: [{ itemIndex: 1, people: ["Pravin", "Wei"] }, { itemIndex: 4, people: ["Pravin", "Wei"] }],
    defaultRule: ex, notes: "",
  }),
  c("two-and-joined-clauses", "Pravin had the nasi goreng and Aisha had the mee goreng, the rest is shared", {
    assignments: [{ itemIndex: 0, people: ["Pravin"] }, { itemIndex: 1, people: ["Aisha"] }], defaultRule: eq, notes: "",
  }),
  c("category-dessert-payer", "Farah is paying for the desserts", {
    assignments: [{ itemIndex: 5, people: ["Farah"] }], defaultRule: eq, notes: "",
  }),
  c("partial-name-roti", "Aisha pays for the roti", {
    assignments: [{ itemIndex: 2, people: ["Aisha"] }], defaultRule: eq, notes: "",
  }),
  c("shared-pair-then-complement-payer", "Pravin and Wei split the mee goreng and nasi goreng, Aisha pays for the rest", {
    assignments: [
      { itemIndex: 0, people: ["Pravin", "Wei"] },
      { itemIndex: 1, people: ["Pravin", "Wei"] },
      { itemIndex: 2, people: ["Aisha"] },
      { itemIndex: 3, people: ["Aisha"] },
      { itemIndex: 4, people: ["Aisha"] },
      { itemIndex: 5, people: ["Aisha"] },
    ],
    defaultRule: eq, notes: "",
  }),
  c("plain-equal", "Split everything equally", { assignments: [], defaultRule: eq, notes: "" }),
  c("two-comma-clauses", "Farah has the milo ais, Wei has the teh tarik", {
    assignments: [{ itemIndex: 4, people: ["Farah"] }, { itemIndex: 3, people: ["Wei"] }], defaultRule: eq, notes: "",
  }),
  c("everyone-except-person", "Everyone except Wei splits the cendol", {
    assignments: [{ itemIndex: 5, people: ["Pravin", "Aisha", "Farah"] }], defaultRule: eq, notes: "",
  }),
  c("two-items-one-person", "Wei had the cendol and the milo ais", {
    assignments: [{ itemIndex: 5, people: ["Wei"] }, { itemIndex: 4, people: ["Wei"] }], defaultRule: eq, notes: "",
  }),
  c("possessive-contraction", "Pravin's paying for the mee goreng", {
    assignments: [{ itemIndex: 1, people: ["Pravin"] }], defaultRule: eq, notes: "",
  }),
  c("pair-shares-two-items", "Aisha and Farah split the nasi goreng and mee goreng", {
    assignments: [{ itemIndex: 0, people: ["Aisha", "Farah"] }, { itemIndex: 1, people: ["Aisha", "Farah"] }],
    defaultRule: eq, notes: "",
  }),
  // --- phrasings a rule-based parser can't fully resolve: a question is fine, a wrong answer is not
  c("synonym-not-in-item-names", "Pravin had the tea", {
    assignments: [{ itemIndex: 3, people: ["Pravin"] }], defaultRule: eq, notes: "",
  }, true),
  { ...c("unknown-person", "Sarah owes for the nasi goreng", {
    assignments: [{ itemIndex: 0, people: ["Sarah"] }], defaultRule: eq, notes: "",
  }, true), mustAsk: true },
  c("typo-in-name", "Pravn had the nasi goreng", {
    assignments: [{ itemIndex: 0, people: ["Pravin"] }], defaultRule: eq, notes: "",
  }, true),
  c("misspelled-item", "Wei had the chendol", {
    assignments: [{ itemIndex: 5, people: ["Wei"] }], defaultRule: eq, notes: "",
  }, true),
  // --- who pays the tax / service charge (added after a user's real receipt exposed the gap)
  ct("tax-payer-food-and-tax", "Pravin pays for the food and Wifey pays the tax", {
    assignments: [{ itemIndex: 0, people: ["Pravin"] }, { itemIndex: 1, people: ["Pravin"] }],
    defaultRule: eq, taxPayers: ["Wifey"], notes: "",
  }),
  ct("tax-payer-service-charge", "Wifey covers the service charge, split the rest equally", {
    assignments: [], defaultRule: eq, taxPayers: ["Wifey"], notes: "",
  }),
  ct("tax-default-equal-cents", "Split equally", { assignments: [], defaultRule: eq, notes: "" }),
];
