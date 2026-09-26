import type { ReceiptItem } from "@/types/domain";
import { classifyItem, normalise } from "@/lib/retrieval/lexicon";

// Deterministic merchant category for history filters — no LLM. A keyword table
// over the merchant name wins; otherwise the dominant item categories decide.

export const MERCHANT_CATEGORIES = [
  "Groceries",
  "Food & Drink",
  "Fuel",
  "Health & Pharmacy",
  "Entertainment",
  "Transport",
  "Other",
] as const;
export type MerchantCategory = (typeof MERCHANT_CATEGORIES)[number];

// Longest / most specific first is not required: a merchant name rarely hits
// two rows, and the table order below puts the ambiguous rows (food) last.
const MERCHANT_KEYWORDS: Array<[MerchantCategory, string[]]> = [
  ["Groceries", [
    "99 speedmart", "speedmart", "mydin", "tesco", "lotus", "giant", "aeon", "econsave",
    "jaya grocer", "village grocer", "mart", "supermarket", "hypermarket", "grocer",
    "family mart", "familymart", "7 eleven", "7eleven", "kk mart", "kkmart",
  ]],
  ["Fuel", ["petronas", "shell", "petron", "caltex", "bhp", "petrol", "fuel"]],
  ["Health & Pharmacy", ["guardian", "watsons", "caring", "pharmacy", "farmasi", "clinic", "klinik"]],
  ["Entertainment", ["cinema", "tgv", "gsc", "karaoke", "bowling", "arcade", "theme park"]],
  ["Transport", ["grab", "uber", "taxi", "bolt", "touch n go", "tng", "parking"]],
  ["Food & Drink", [
    "restaurant", "restoran", "cafe", "coffee", "kopitiam", "mamak", "bistro", "bar", "pub",
    "grill", "kitchen", "mcd", "mcdonald", "kfc", "starbucks", "pizza", "domino", "secret recipe",
    "old town", "nando", "sushi", "warung", "nasi", "bakery", "dapur", "steakhouse",
  ]],
];

const FOOD_CATEGORY_IDS = new Set(["drinks", "desserts", "alcohol", "mains", "sides"]);

function hasKeyword(haystack: string, kw: string): boolean {
  // Whole-word/phrase match on normalised text, so "bar" doesn't hit "barber".
  return ` ${haystack} `.includes(` ${normalise(kw)} `);
}

export function inferMerchantCategory(
  merchantName: string | null | undefined,
  items: ReceiptItem[],
): MerchantCategory {
  const name = normalise(merchantName ?? "");
  if (name) {
    for (const [category, keywords] of MERCHANT_KEYWORDS) {
      if (keywords.some((kw) => hasKeyword(name, kw))) return category;
    }
  }

  // No usable merchant name: what was actually ordered. If most lines look like
  // food or drink, it's Food & Drink; otherwise we don't guess.
  if (items.length > 0) {
    const food = items.filter((it) => {
      const top = classifyItem(it.name)[0];
      return top !== undefined && FOOD_CATEGORY_IDS.has(top.categoryId);
    }).length;
    if (food / items.length >= 0.5) return "Food & Drink";
  }
  return "Other";
}
