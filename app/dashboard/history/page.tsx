import { createClient } from "@/utils/supabase/server";
import HistoryList from "@/components/history-list";

export const dynamic = 'force-dynamic';

export default async function HistoryPage() {
  const supabase = await createClient();
  
  const { data: history } = await supabase
    .from('bill_history')
    .select('*')
    .order('created_at', { ascending: false });

  return (
    <div className="p-6 md:p-10 max-w-4xl mx-auto space-y-8 mb-20">
      <header className="space-y-1">
        <h1 className="text-3xl font-black tracking-tighter uppercase italic text-white">History</h1>
        <p className="text-zinc-500 text-xs font-mono uppercase tracking-widest">Past Settlements</p>
      </header>

      <HistoryList initialHistory={history || []} />
    </div>
  );
}