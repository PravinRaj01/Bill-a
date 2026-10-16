import { formatMoney } from "@/lib/money";
import { settleStatus, type BillSettleSummary } from "@/lib/split/settle";

/**
 * Settle-up state of one bill as a short pill, for History and Home. `today` is the user's own date (the
 * server's is not), so overdue and due-in-N-days are worked out in the browser. Null for an untracked bill.
 */
export function settlePill(entry: { settle: BillSettleSummary | null; currency?: string | null }, today: string): { text: string; tone: string } | null {
  const s = entry.settle;
  if (!s) return null;
  const cur = entry.currency || "RM";
  if (s.mismatch) return { text: "Check who paid", tone: "border-amber-500/40 bg-amber-500/10 text-amber-300" };
  const { status, dueIn } = settleStatus(s.outstanding, s.dueDate, today);
  const left = `${formatMoney(s.outstanding, cur)} outstanding`;
  if (status === "settled") return { text: "Settled ✓", tone: "border-success/40 bg-success/10 text-success" };
  if (status === "overdue") return { text: `${left} · overdue ${-dueIn!} ${dueIn === -1 ? "day" : "days"}`, tone: "border-coral/40 bg-coral/10 text-coral" };
  if (status === "due") return { text: `${left} · due today`, tone: "border-amber-500/40 bg-amber-500/10 text-amber-300" };
  if (dueIn !== null) return { text: `${left} · due in ${dueIn} ${dueIn === 1 ? "day" : "days"}`, tone: "border-white/10 bg-white/5 text-zinc-300" };
  return { text: `${left} · ${s.paidCount} of ${s.transferCount} paid`, tone: "border-white/10 bg-white/5 text-zinc-300" };
}
