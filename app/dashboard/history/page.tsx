import HistoryList from "@/components/history-list";
import { listBills } from "@/lib/actions/history";

export const dynamic = 'force-dynamic';

export default async function HistoryPage() {
  // listBills() resolves the user from the verified session and scopes the SQL
  // to it. This page used to be a bare `select('*')` that relied entirely on
  // Supabase RLS — on plain Postgres that would have shown every user's history.
  const history = await listBills();

  return (
    <div className="p-6 md:p-10 max-w-4xl mx-auto space-y-8 mb-20">
      <header className="space-y-1">
        <h1 className="text-3xl font-black tracking-tighter uppercase italic text-white">History</h1>
        <p className="text-zinc-500 text-xs font-mono uppercase tracking-widest">Past Settlements</p>
      </header>

      <HistoryList initialHistory={history} />
    </div>
  );
}
