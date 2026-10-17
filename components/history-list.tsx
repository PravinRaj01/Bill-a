"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { ArrowRight, Calendar, Check, Receipt } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ConfirmSheet } from "@/components/ui/confirm-sheet";
import { Segmented } from "@/components/ui/segmented";
import { clearHistory, deleteBills, type BillListEntry } from "@/lib/actions/history";
import { formatMoney } from "@/lib/money";
import { localDate } from "@/lib/split/settle";
import { settlePill } from "@/lib/ui/settle-pill";
import { cn } from "@/lib/utils";

type Filter = "all" | "open" | "settled";

const isUnsettled = (e: BillListEntry) => !!e.settle && (e.settle.mismatch || e.settle.outstanding > 0);
const isSettled = (e: BillListEntry) => !!e.settle && !e.settle.mismatch && e.settle.outstanding === 0;

export default function HistoryList({ initialHistory }: { initialHistory: BillListEntry[] }) {
  const [history, setHistory] = useState(initialHistory);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [isEditMode, setIsEditMode] = useState(false);
  const [filter, setFilter] = useState<Filter>("all");
  const [confirm, setConfirm] = useState<null | "selected" | "all">(null);
  // The date-dependent parts (overdue, due in N days) wait for the browser: the server's "today" can differ.
  const [today, setToday] = useState<string | null>(null);
  const router = useRouter();

  useEffect(() => {
    setToday(localDate());
  }, []);

  // Sync state if parent passes new data (Fixes "History not updating")
  useEffect(() => {
    setHistory(initialHistory);
  }, [initialHistory]);

  const counts = useMemo(
    () => ({ all: history.length, open: history.filter(isUnsettled).length, settled: history.filter(isSettled).length }),
    [history],
  );
  const shown = history.filter((e) => (filter === "open" ? isUnsettled(e) : filter === "settled" ? isSettled(e) : true));

  // What people still owe you, and what you still owe, summed per currency over bills you've tracked.
  const owed = useMemo(() => {
    const byCurrency = new Map<string, { owedToYou: number; youOwe: number }>();
    for (const e of history) {
      if (!e.settle?.mine) continue;
      const cur = e.currency || "RM";
      const t = byCurrency.get(cur) ?? { owedToYou: 0, youOwe: 0 };
      t.owedToYou += e.settle.mine.owedToYou;
      t.youOwe += e.settle.mine.youOwe;
      byCurrency.set(cur, t);
    }
    return [...byCurrency].filter(([, t]) => t.owedToYou > 0 || t.youOwe > 0);
  }, [history]);

  const toggleSelect = (id: string) => {
    setSelectedIds((prev) => (prev.includes(id) ? prev.filter((i) => i !== id) : [...prev, id]));
  };

  const deleteSelected = async () => {
    // The server only deletes ids that belong to the signed-in user; any other
    // id in this list is silently ignored, so this can't touch anyone else's data.
    try {
      await deleteBills(selectedIds);
      setHistory((prev) => prev.filter((item) => !selectedIds.includes(item.id)));
      toast(`Deleted ${selectedIds.length} ${selectedIds.length === 1 ? "item" : "items"}`);
      setSelectedIds([]);
      setIsEditMode(false);
      router.refresh();
    } catch (e) {
      console.error("Failed to delete", e);
      toast.error("Couldn't delete those items. Please try again.");
    }
  };

  const deleteAll = async () => {
    try {
      await clearHistory();
      setHistory([]);
      setIsEditMode(false);
      toast("History cleared");
      router.refresh();
    } catch (e) {
      console.error("Failed to clear history", e);
      toast.error("Couldn't clear history. Please try again.");
    }
  };

  return (
    <div className="space-y-4">
      {owed.length > 0 && (
        <div data-testid="owed-summary" className="space-y-1 rounded-2xl glass p-4">
          {owed.map(([cur, t]) => (
            <div key={cur} className="space-y-0.5 text-sm text-muted-foreground">
              {t.owedToYou > 0 && (
                <p>
                  You&apos;re owed <strong className="text-lg font-bold text-success tabular-nums">{formatMoney(t.owedToYou, cur)}</strong>
                </p>
              )}
              {t.youOwe > 0 && (
                <p>
                  You owe <strong className="text-lg font-bold text-coral tabular-nums">{formatMoney(t.youOwe, cur)}</strong>
                </p>
              )}
            </div>
          ))}
        </div>
      )}

      {counts.open + counts.settled > 0 && (
        <Segmented<Filter>
          label="Filter bills"
          value={filter}
          onValueChange={setFilter}
          options={[
            { value: "all", label: `All ${counts.all}` },
            { value: "open", label: `Unsettled ${counts.open}` },
            { value: "settled", label: `Settled ${counts.settled}` },
          ]}
        />
      )}

      {/* Toolbar */}
      <div className="flex items-center justify-between rounded-2xl glass p-2">
        <div className="flex gap-2">
          {!isEditMode ? (
            <Button variant="ghost" size="sm" onClick={() => setIsEditMode(true)} className="h-10 text-xs font-semibold text-muted-foreground hover:text-foreground">
              Edit list
            </Button>
          ) : (
            <>
              <Button variant="destructive" size="sm" onClick={() => setConfirm("selected")} disabled={selectedIds.length === 0} className="h-10 rounded-lg text-xs font-semibold">
                Delete ({selectedIds.length})
              </Button>
              <Button variant="ghost" size="sm" onClick={() => { setIsEditMode(false); setSelectedIds([]); }} className="h-10 text-xs font-semibold text-muted-foreground hover:text-foreground">
                Cancel
              </Button>
            </>
          )}
        </div>
        {isEditMode && (
          <Button variant="link" onClick={() => setConfirm("all")} className="h-10 text-xs font-semibold text-destructive hover:text-destructive">
            Clear all
          </Button>
        )}
      </div>

      <div className="space-y-3">
        {shown.length > 0 ? (
          shown.map((bill) => {
            const pill = today ? settlePill(bill, today) : null;
            const s = bill.settle;
            const cur = bill.currency || "RM";
            return (
              <div key={bill.id} className="flex items-center gap-3 animate-in slide-in-from-bottom-2">
                {isEditMode && (
                  <button
                    type="button"
                    aria-label={selectedIds.includes(bill.id) ? `Deselect ${bill.billTitle}` : `Select ${bill.billTitle}`}
                    onClick={() => toggleSelect(bill.id)}
                    className="flex size-11 shrink-0 items-center justify-center"
                  >
                    {selectedIds.includes(bill.id) ? (
                      <span className="flex size-6 items-center justify-center rounded-md bg-primary"><Check className="size-4 text-primary-foreground" strokeWidth={4} /></span>
                    ) : (
                      <span className="size-6 rounded-md border-2 border-white/15" />
                    )}
                  </button>
                )}
                <Card className={cn("group flex-1 gap-0 overflow-hidden rounded-2xl border-white/8 glass py-0", selectedIds.includes(bill.id) && "opacity-50")}>
                  <CardContent className="p-0">
                    <Link
                      href={`/dashboard/history/${bill.id}`}
                      transitionTypes={["nav-forward"]}
                      data-testid="bill-row"
                      className="block space-y-3 p-5"
                      onClick={(e) => isEditMode && e.preventDefault()}
                    >
                      <div className="flex items-center gap-4">
                        <div className="flex size-10 shrink-0 items-center justify-center rounded-full border border-white/8 bg-white/5">
                          <Receipt className="size-4 text-muted-foreground" />
                        </div>
                        <div className="min-w-0 flex-1">
                          <h3 className="truncate text-base font-bold text-foreground">{bill.billTitle}</h3>
                          <div className="mt-1 space-y-0.5 text-xs text-muted-foreground">
                            {/* toISOString() on purpose: the server and the browser must print the same date (hydration) */}
                            <p className="flex items-center gap-1.5 whitespace-nowrap">
                              <Calendar className="size-3" /> {new Date(bill.createdAt).toISOString().split("T")[0]}
                            </p>
                            {s && s.payers.length > 0 && (
                              <p data-testid="paid-by">Paid by {s.payers.length === 1 ? s.payers[0] : `${s.payers.length} people`}</p>
                            )}
                          </div>
                        </div>
                        <div className="shrink-0 text-right">
                          <div className="text-base font-bold tabular-nums text-foreground">{formatMoney(bill.totalAmount || 0, cur)}</div>
                          {!isEditMode && (
                            <div className="flex items-center justify-end gap-1 text-xs font-semibold text-muted-foreground transition-colors group-hover:text-foreground">
                              Details <ArrowRight className="size-3" />
                            </div>
                          )}
                        </div>
                      </div>

                      {pill && s && (
                        <div className="space-y-2">
                          <span data-testid="settle-pill" className={cn("inline-flex min-h-7 items-center rounded-full border px-3 text-xs font-semibold", pill.tone)}>
                            {pill.text}
                          </span>
                          {!s.mismatch && s.owed > 0 && (
                            <div
                              role="progressbar"
                              aria-label="Collected"
                              aria-valuemin={0}
                              aria-valuemax={s.owed}
                              aria-valuenow={s.collected}
                              className="h-1.5 overflow-hidden rounded-full bg-white/10"
                            >
                              <div className="h-full rounded-full bg-success" style={{ width: `${(s.collected / s.owed) * 100}%` }} />
                            </div>
                          )}
                        </div>
                      )}
                    </Link>
                  </CardContent>
                </Card>
              </div>
            );
          })
        ) : (
          <div className="rounded-3xl border-2 border-dashed border-white/8 py-20 text-center">
            <p className="text-sm text-muted-foreground">
              {history.length === 0 ? "No history yet." : filter === "open" ? "Nothing unsettled. Everyone has paid." : "No settled bills yet."}
            </p>
          </div>
        )}
      </div>

      <ConfirmSheet
        open={confirm === "selected"}
        onOpenChange={(o) => !o && setConfirm(null)}
        title={`Delete ${selectedIds.length} ${selectedIds.length === 1 ? "item" : "items"}?`}
        description="This removes them from your history. It can't be undone."
        confirmLabel="Delete"
        onConfirm={deleteSelected}
      />
      <ConfirmSheet
        open={confirm === "all"}
        onOpenChange={(o) => !o && setConfirm(null)}
        title="Delete everything?"
        description="Every saved bill is removed from your history. This can't be undone."
        confirmLabel="Delete everything"
        onConfirm={deleteAll}
      />
    </div>
  );
}
