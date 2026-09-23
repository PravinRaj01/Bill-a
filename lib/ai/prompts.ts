// Centralized prompt construction for the assignment-plan LLM call, used
// by both app/dev/webllm-spike and app/dev/model-bakeoff so the two
// harnesses never silently drift apart.
//
// Revision history worth keeping in mind:
// - Round 1 (5 models x 15 cases): a `weights` field in the schema let
//   Llama-3.2-1B and Qwen2.5-1.5B — two different model families —
//   independently hallucinate weights that mirror the item's receipt
//   `quantity` field rather than a real split ratio, even on the
//   simplest possible case ("split equally", one item, qty 3 -> both
//   invented [3,1,1]/[3,0,0] instead of an even split).
// - Round 2: added an explicit worked counter-example targeting exactly
//   that pattern. Result: byte-identical failure, same item, same
//   instruction. Conclusion: not a prompting problem at this model size
//   — `weights` was removed from the schema entirely (lib/ai/schemas.ts)
//   rather than iterated on further.
// - Round 3 (8 models): with weights gone, 5 of 15 cases STILL failed on
//   every model 1.5B-3.8B — all of them cases requiring the model to
//   search the raw item list for a category ("the drinks") or compute a
//   complement ("the rest", "except X"). Scaling 1.5B->3.8B did not touch
//   this cluster. Conclusion: this is a search/set-arithmetic task, which
//   small LLMs are structurally bad at and plain code is perfect at — see
//   lib/retrieval/resolve.ts (plan §5.1b). `resolvedBlock` below is where
//   that layer's output gets injected, turning "search the list" into
//   "confirm this already-resolved list" — the single-clause form
//   Phase 1 measured passing 7/9 of the time.
export function buildSystemPrompt(candidateIndices: number[], peopleNames: string[]): string {
  return (
    "You assign receipt items to people. You never calculate money.\n" +
    "Respond with a single JSON object matching exactly:\n" +
    '{"assignments":[{"itemIndex":<int>,"people":["<name>",...]}],' +
    '"defaultRule":"equal"|"exclude","notes":"<string>"}\n' +
    `itemIndex must be one of ${JSON.stringify(candidateIndices)}. ` +
    `people must be drawn only from ${JSON.stringify(peopleNames)}, no duplicates. ` +
    "Every person named in the instruction must appear in the people list; ignore unknown names. " +
    "Every person you list for an item shares it EVENLY — there is no way to give someone a " +
    "larger or smaller share of a single item. " +
    "Items not mentioned follow defaultRule. Emit each item index at most once.\n" +
    "\n" +
    "When deciding which items match a category the user names (e.g. \"drinks\", \"desserts\"), " +
    "judge by the item's own name — common drink words include tea, coffee, juice, soda, " +
    "milkshake, lemonade; common dessert words include cake, sorbet, ice cream, pudding — and " +
    "double-check you assigned the category to the PERSON who was named as paying for it, not " +
    "the opposite.\n" +
    "\n" +
    "If a RESOLVED REFERENCES block is given in the user message, those index lists have " +
    "already been worked out correctly — use them directly instead of re-deriving which items " +
    "a phrase like \"the drinks\" or \"the rest\" refers to. Only figure out the mapping yourself " +
    "for phrases NOT covered by that block."
  );
}

export function buildUserPrompt(
  people: string[],
  menu: string,
  instruction: string,
  resolvedBlock?: string,
): string {
  const resolved = resolvedBlock ? `\n\n${resolvedBlock}` : "";
  return `PEOPLE: ${JSON.stringify(people)}\n\nCANDIDATE ITEMS (index, name, quantity):\n${menu}\n\nINSTRUCTION: ${JSON.stringify(instruction)}${resolved}`;
}

export function buildItemMenu(items: Array<{ name: string; quantity: number }>): string {
  return items.map((item, i) => `  [${i}] ${item.name} (qty ${item.quantity})`).join("\n");
}
