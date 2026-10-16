"use client";
import { useMemo, useState } from "react";

import { canMove, initialState, maxFor, shareReducer, type ShareAction, type ShareState } from "@/lib/split/sliders";
import type { Cents } from "@/types/domain";

// React wrapper over the pure share maths in lib/split/sliders.ts. The bill (`total`) can change
// under the editor (an item or the tax is edited, the tax switch flips), so the stored state is
// always viewed through rescale(): proportions stay, the total follows. No effects, no extra
// renders. Key the component using this by the people list: a different group starts equal.

export function useShareEditor(people: string[], total: Cents, seed?: Cents[]) {
  const [stored, setStored] = useState<ShareState>(() => initialState(total, people.length, seed));
  const view = useMemo(() => shareReducer(stored, { type: "rescale", total }), [stored, total]);

  const apply = (action: ShareAction) => setStored(shareReducer(view, action));

  return {
    amounts: view.amounts,
    locked: view.locked,
    total,
    canMove: (i: number) => canMove(view, i),
    maxFor: (i: number) => maxFor(view, i),
    drag: (i: number, cents: Cents) => apply({ type: "drag", i, cents }),
    type: (i: number, cents: Cents) => apply({ type: "type", i, cents }),
    toggleLock: (i: number) => apply({ type: "lock", i }),
    /** Back to equal shares. Returns a function that undoes it. */
    reset: () => {
      const before = view;
      apply({ type: "equal" });
      return () => setStored(before);
    },
  };
}

export type ShareEditor = ReturnType<typeof useShareEditor>;
