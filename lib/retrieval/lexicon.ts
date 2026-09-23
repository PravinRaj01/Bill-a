// Category lexicon — tier 1 of the resolution cascade (plan §5.1b).
//
// Why this exists: the Phase 1 bake-off showed EVERY model from 1.5B to
// 3.8B failing to work out which items "the drinks" or "the desserts"
// refers to. That is a lookup, not a reasoning task, and a word list does
// it perfectly and inspectably. Tier 1 catches the common cases at zero
// cost; anything it can't classify falls through to embeddings (tier 2)
// and then to a per-item binary LLM question (tier 3).
//
// Localisation is where the real accuracy lives: a Malaysian receipt's
// "drinks" are teh tarik, kopi o, milo ais, sirap bandung — none of which
// a generic English word list would catch.

export interface Category {
  id: string;
  /** Tokens which, appearing in an ITEM name, indicate this category. */
  itemKeywords: string[];
  /** Phrases a USER might type to refer to this category. */
  queryPhrases: string[];
}

export const CATEGORIES: Category[] = [
  {
    id: "drinks",
    itemKeywords: [
      // generic
      "tea", "coffee", "juice", "soda", "cola", "coke", "pepsi", "sprite",
      "water", "lemonade", "milkshake", "shake", "smoothie", "latte",
      "cappuccino", "espresso", "mocha", "americano", "frappe", "cordial",
      "soft drink", "softdrink", "beverage", "drink",
      // malaysian / SEA — the ones a generic list would miss
      "teh", "tarik", "kopi", "milo", "sirap", "bandung", "limau",
      "barli", "cincau", "horlicks", "nescafe", "neslo", "ais kacang",
      "100 plus", "100plus", "ribena", "sky juice",
    ],
    queryPhrases: [
      "drinks", "drink", "beverages", "beverage", "the drinks",
      "soft drinks", "cold drinks", "hot drinks", "anything to drink",
    ],
  },
  {
    id: "desserts",
    itemKeywords: [
      "cake", "sorbet", "ice cream", "icecream", "gelato", "pudding",
      "brownie", "cheesecake", "tiramisu", "waffle", "pancake", "crepe",
      "tart", "pie", "mousse", "sundae", "parfait", "dessert",
      // SEA
      "cendol", "bubur", "kuih", "apam", "onde", "pisang goreng",
    ],
    queryPhrases: [
      "desserts", "dessert", "the desserts", "sweets", "sweet stuff",
      "pudding", "anything sweet",
    ],
  },
  {
    id: "alcohol",
    itemKeywords: [
      "beer", "wine", "whisky", "whiskey", "vodka", "gin", "rum",
      "tequila", "cocktail", "mojito", "margarita", "sake", "soju",
      "cider", "stout", "lager", "ale", "tiger", "heineken", "carlsberg",
    ],
    queryPhrases: [
      "alcohol", "drinks with alcohol", "booze", "beers", "the beers",
      "alcoholic drinks", "the alcohol",
    ],
  },
  {
    id: "mains",
    itemKeywords: [
      "rice", "noodle", "noodles", "pasta", "spaghetti", "burger", "pizza",
      "steak", "chicken", "beef", "lamb", "fish", "curry", "set", "meal",
      "sandwich", "wrap", "roll", "platter", "grill", "grilled", "fried",
      // SEA
      "nasi", "mee", "kuey", "teow", "laksa", "rendang", "goreng",
      "roti", "canai", "murtabak", "satay", "char", "bihun", "maggi",
    ],
    queryPhrases: [
      "mains", "main course", "main courses", "the mains", "food",
      "the food", "meals", "entrees", "main dishes",
    ],
  },
  {
    id: "sides",
    itemKeywords: [
      "fries", "chips", "salad", "soup", "bread", "garlic bread", "nuggets",
      "wings", "spring roll", "springroll", "dumpling", "side", "coleslaw",
      "mash", "wedges", "papadom", "keropok", "acar",
    ],
    queryPhrases: [
      "sides", "side dishes", "the sides", "starters", "appetisers",
      "appetizers", "the starters", "snacks",
    ],
  },
];

function normalise(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9\s]/g, " ").replace(/\s+/g, " ").trim();
}

/**
 * Scores an item name against every category by counting keyword hits.
 * Multi-word keywords ("ice cream", "garlic bread") are matched as
 * phrases so they beat single-word matches from another category — which
 * is what stops "CHOCOLATE LAVA CAKE" being classified as a drink on the
 * strength of "chocolate", or "MILKSHAKE" landing in desserts.
 */
export function classifyItem(itemName: string): { categoryId: string; score: number }[] {
  const haystack = normalise(itemName);
  const tokens = new Set(haystack.split(" "));

  return CATEGORIES.map((cat) => {
    let score = 0;
    for (const kw of cat.itemKeywords) {
      if (kw.includes(" ")) {
        // Phrase keyword — weight higher; it's more specific evidence.
        if (haystack.includes(kw)) score += 2;
      } else if (tokens.has(kw)) {
        score += 1;
      }
    }
    return { categoryId: cat.id, score };
  })
    .filter((r) => r.score > 0)
    .sort((a, b) => b.score - a.score);
}

/**
 * Finds which category (if any) a user's phrase is referring to.
 * Longest phrase match wins, so "hot drinks" beats a bare "drinks".
 */
export function matchCategoryPhrase(phrase: string): string | null {
  const needle = normalise(phrase);
  let best: { id: string; len: number } | null = null;

  for (const cat of CATEGORIES) {
    for (const qp of cat.queryPhrases) {
      const n = normalise(qp);
      if (needle === n || needle.includes(n)) {
        if (!best || n.length > best.len) best = { id: cat.id, len: n.length };
      }
    }
  }
  return best?.id ?? null;
}

export { normalise };
