import HistoryList from "@/components/history-list";
import { HiddenBills } from "@/components/history/HiddenBills";
import { SharedList } from "@/components/history/SharedList";
import { PageTransition } from "@/components/page-transition";
import { listBills, listHiddenBills } from "@/lib/actions/history";
import { listSharedWithMe } from "@/lib/actions/shared";

export const dynamic = 'force-dynamic';

export default async function HistoryPage() {
  // listBills() resolves the user from the verified session and scopes the SQL
  // to it. This page used to be a bare `select('*')` that relied entirely on
  // Supabase RLS — on plain Postgres that would have shown every user's history.
  const [history, shared, hidden] = await Promise.all([listBills(), listSharedWithMe(true), listHiddenBills()]);

  return (
    <PageTransition>
    <div className="p-6 md:p-10 max-w-4xl mx-auto space-y-8 mb-20">
      <header className="space-y-1">
        <h1 className="type-title">History</h1>
        <p className="text-muted-foreground text-xs font-mono uppercase tracking-widest">Past Settlements</p>
      </header>

      <SharedList initial={shared} />

      <HistoryList initialHistory={history} />

      <HiddenBills initial={hidden} />
    </div>
    </PageTransition>
  );
}
