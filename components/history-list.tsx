"use client";

import { useState, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Receipt, Calendar, ArrowRight, Check } from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { createClient } from "@/utils/supabase/client";
import { useRouter } from "next/navigation";

export default function HistoryList({ initialHistory }: { initialHistory: any[] }) {
  const [history, setHistory] = useState(initialHistory);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [isEditMode, setIsEditMode] = useState(false);
  const supabase = createClient();
  const router = useRouter();

  // Sync state if parent passes new data (Fixes "History not updating")
  useEffect(() => {
    setHistory(initialHistory);
  }, [initialHistory]);

  const toggleSelect = (id: string) => {
    setSelectedIds(prev => prev.includes(id) ? prev.filter(i => i !== id) : [...prev, id]);
  };

  const deleteSelected = async () => {
    if (selectedIds.length === 0) return;
    if (!confirm(`Delete ${selectedIds.length} items?`)) return;

    const { error } = await supabase.from('bill_history').delete().in('id', selectedIds);
    if (!error) {
      setHistory(prev => prev.filter(item => !selectedIds.includes(item.id)));
      setSelectedIds([]);
      setIsEditMode(false);
      router.refresh();
    }
  };

  const deleteAll = async () => {
    if (!confirm("Delete EVERYTHING? This cannot be undone.")) return;
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) return;

    const { error } = await supabase.from('bill_history').delete().eq('user_id', user.id);
    if (!error) {
      setHistory([]);
      setIsEditMode(false);
      router.refresh();
    }
  };

  return (
    <div className="space-y-4">
      {/* Toolbar */}
      <div className="flex justify-between items-center bg-zinc-900/30 p-2 rounded-2xl border border-white/5">
        <div className="flex gap-2">
          {!isEditMode ? (
            <Button variant="ghost" size="sm" onClick={() => setIsEditMode(true)} className="text-[10px] font-black uppercase text-zinc-500 hover:text-white">
              Edit List
            </Button>
          ) : (
            <>
              <Button variant="destructive" size="sm" onClick={deleteSelected} disabled={selectedIds.length === 0} className="text-[10px] font-black uppercase rounded-lg h-8">
                Delete ({selectedIds.length})
              </Button>
              <Button variant="ghost" size="sm" onClick={() => { setIsEditMode(false); setSelectedIds([]); }} className="text-[10px] font-black uppercase text-zinc-400 hover:text-white h-8">
                Cancel
              </Button>
            </>
          )}
        </div>
        {isEditMode && (
          <Button variant="link" onClick={deleteAll} className="text-[10px] font-black uppercase text-red-600 hover:text-red-500 h-8">
            Clear All
          </Button>
        )}
      </div>

      <div className="space-y-3">
        {history.length > 0 ? (
           history.map((bill) => (
            <div key={bill.id} className="flex items-center gap-3 animate-in slide-in-from-bottom-2">
              {isEditMode && (
                <button onClick={() => toggleSelect(bill.id)} className="transition-all shrink-0">
                  {selectedIds.includes(bill.id) ? 
                    <div className="w-6 h-6 bg-white rounded-md flex items-center justify-center"><Check className="text-black w-4 h-4" strokeWidth={4} /></div> : 
                    <div className="w-6 h-6 border-2 border-zinc-800 rounded-md" />
                  }
                </button>
              )}
              <Card className={`flex-1 bg-[#0c0c0e] border-white/5 rounded-2xl group overflow-hidden ${selectedIds.includes(bill.id) ? 'opacity-50' : ''}`}>
                <CardContent className="p-0">
                  <Link href={`/dashboard/history/${bill.id}`} className="flex items-center p-5 gap-4" onClick={(e) => isEditMode && e.preventDefault()}>
                    <div className="w-10 h-10 bg-white/5 rounded-full flex items-center justify-center border border-white/5 shrink-0">
                      <Receipt className="w-4 h-4 text-zinc-500" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <h3 className="font-bold text-zinc-100 uppercase tracking-tight truncate text-sm">{bill.bill_title}</h3>
                      <div className="flex items-center gap-2 text-[9px] text-zinc-600 font-mono mt-1 uppercase">
                        {/* FIX: Use toISOString() to prevent Hydration Mismatch Error */}
                        <Calendar className="w-3 h-3" /> {new Date(bill.created_at).toISOString().split('T')[0]}
                      </div>
                    </div>
                    <div className="text-right shrink-0">
                      <div className="font-mono font-bold text-white text-sm">
                        {/* FIX: Safety check for null amounts */}
                        {bill.currency || 'RM'}{(bill.total_amount || 0).toFixed(2)}
                      </div>
                      {!isEditMode && <div className="text-[9px] text-zinc-700 uppercase font-black flex items-center justify-end gap-1 group-hover:text-white transition-colors">Details <ArrowRight className="w-3 h-3" /></div>}
                    </div>
                  </Link>
                </CardContent>
              </Card>
            </div>
          ))
        ) : (
          <div className="py-20 text-center border-2 border-dashed border-zinc-900 rounded-3xl">
             <p className="text-zinc-600 font-medium italic text-xs">No history yet.</p>
          </div>
        )}
      </div>
    </div>
  );
}