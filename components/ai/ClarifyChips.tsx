"use client";

import { HelpCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { Chip } from "@/lib/split/fallback-parser";

// "Did you mean…?" — the questions the router and fallback parser raise instead of
// guessing. Each button either rewrites the instruction and re-runs the split, adds
// a person, or (as a last resort) goes ahead with the best guess.

export type ClarifyAction =
  | { type: "replace"; from: string; to: string }
  | { type: "add-person"; name: string }
  | { type: "continue-anyway" }
  | { type: "use-preview" }
  | { type: "edit" };

const pill =
  "h-8 rounded-full border border-white/10 bg-white/5 px-3 text-[11px] font-bold text-white hover:bg-white/10 hover:text-white";

export function ClarifyChips({
  chips,
  hasPreview,
  onAction,
}: {
  chips: Chip[];
  hasPreview: boolean;
  onAction: (a: ClarifyAction) => void;
}) {
  return (
    <div data-testid="clarify" className="space-y-3 rounded-2xl border border-indigo-500/30 bg-indigo-500/5 p-4">
      <div className="flex items-center gap-2 text-indigo-200">
        <HelpCircle className="h-4 w-4" />
        <p className="text-xs font-bold">Quick check before I split this</p>
      </div>

      {chips.map((chip, i) => (
        <div key={i} className="space-y-2 text-xs text-zinc-300" data-testid="clarify-chip" data-kind={chip.kind}>
          {chip.kind === "possible-typo" && (
            <>
              <p>
                You wrote “{chip.token}”. Did you mean <strong>{chip.suggestion}</strong>?
              </p>
              <div className="flex flex-wrap gap-2">
                <Button type="button" variant="ghost" className={pill} onClick={() => onAction({ type: "replace", from: chip.token, to: chip.suggestion })}>
                  Yes, {chip.suggestion}
                </Button>
                <Button type="button" variant="ghost" className={pill} onClick={() => onAction({ type: "add-person", name: chip.token })}>
                  No, add “{chip.token}” as a new person
                </Button>
              </div>
            </>
          )}
          {chip.kind === "unknown-person" && (
            <>
              <p>
                “{chip.token}” isn&apos;t in this group.
              </p>
              <Button type="button" variant="ghost" className={pill} onClick={() => onAction({ type: "add-person", name: chip.token })}>
                Add {chip.token}
              </Button>
            </>
          )}
          {chip.kind === "ambiguous-item" && (
            <>
              <p>
                “{chip.phrase}” could be more than one item. Which one?
              </p>
              <div className="flex flex-wrap gap-2">
                {chip.options.map((o) => (
                  <Button key={o.index} type="button" variant="ghost" className={pill} onClick={() => onAction({ type: "replace", from: chip.phrase, to: o.name })}>
                    {o.name}
                  </Button>
                ))}
              </div>
            </>
          )}
          {chip.kind === "unresolved" && (
            <p>
              {chip.reason}: <span className="italic">“{chip.clause}”</span>. Try naming the item as it appears on the receipt.
            </p>
          )}
          {chip.kind === "exception-unassigned" && (
            <p>
              Who pays for {chip.options.map((o) => o.name).join(" / ")}? Say it like “…, which is just for Sarah”.
            </p>
          )}
        </div>
      ))}

      <div className="flex flex-wrap gap-2 border-t border-white/5 pt-3">
        <Button type="button" variant="ghost" className={pill} onClick={() => onAction({ type: "edit" })}>
          Edit my instruction
        </Button>
        {hasPreview ? (
          <Button type="button" variant="ghost" className={pill} onClick={() => onAction({ type: "use-preview" })}>
            Split anyway (unclear parts shared equally)
          </Button>
        ) : (
          <Button type="button" variant="ghost" className={pill} onClick={() => onAction({ type: "continue-anyway" })}>
            Continue anyway
          </Button>
        )}
      </div>
    </div>
  );
}
