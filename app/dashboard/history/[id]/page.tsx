"use client";

import { useEffect, useState } from "react";
import { createClient } from "@/utils/supabase/client";
import { useRouter, useParams } from "next/navigation"; // Import useParams
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableRow } from "@/components/ui/table";
import { ChevronLeft, Calendar, Trash2, Play, Share2, Loader2 } from "lucide-react";

export default function BillDetailPage() {
  const params = useParams(); // <--- FIX: Get ID from the hook
  const id = params?.id as string; // Safely extract string ID

  const [bill, setBill] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const supabase = createClient();
  const router = useRouter();

  // 1. Fetch Data Client-Side
  useEffect(() => {
    if (!id) return; // Wait until ID is available

    const fetchBill = async () => {
      const { data, error } = await supabase
        .from('bill_history')
        .select('*')
        .eq('id', id)
        .single();
      
      if (error || !data) {
          console.error("Error fetching bill:", error);
          router.push('/dashboard/history');
      } else {
          setBill(data);
      }
      setLoading(false);
    };
    fetchBill();
  }, [id, supabase, router]);

  // 2. Handle Delete
  const handleDelete = async () => {
      const confirm = window.confirm("Are you sure you want to delete this history?");
      if (!confirm) return;

      await supabase.from('bill_history').delete().eq('id', id);
      router.push('/dashboard/history');
      router.refresh();
  };

  // 3. Handle Continue Session
  const handleContinue = () => {
      if (!bill) return;
      
      // Save data to session storage to pass it to the main app
      sessionStorage.setItem("billa_restore_data", JSON.stringify({
          data: bill.data, // This contains { split, items, people } if it's a new save
          currency: bill.currency
      }));
      
      // Redirect to main app with restore flag
      router.push("/dashboard/new?restore_from_history=true");
  };

  if (loading) return <div className="flex h-screen items-center justify-center bg-black text-white"><Loader2 className="animate-spin" /></div>;
  if (!bill) return null;

  // Handle both Old (Array) and New (Object) data formats
  const splitData = Array.isArray(bill.data) 
      ? bill.data 
      : (bill.data.split || bill.data.splits || []);

  const symbol = bill.currency || "RM";

  return (
    <div className="p-6 md:p-10 max-w-2xl mx-auto space-y-6 mb-20 animate-in fade-in">
      <div className="flex justify-between items-center">
        <button onClick={() => router.back()} className="flex items-center text-[10px] text-zinc-500 hover:text-white transition-colors uppercase font-black tracking-widest">
          <ChevronLeft className="w-4 h-4 mr-1" /> Back
        </button>
        <button onClick={handleDelete} className="text-zinc-800 hover:text-red-500 transition-colors p-2">
            <Trash2 className="w-5 h-5" />
        </button>
      </div>

      <header className="flex justify-between items-start gap-4">
        <div className="space-y-1">
          <h1 className="text-4xl font-black tracking-tighter uppercase italic text-white leading-none">{bill.bill_title}</h1>
          <div className="flex items-center gap-2 text-[10px] text-zinc-600 font-mono tracking-widest uppercase mt-2">
            <Calendar className="w-3 h-3" /> {new Date(bill.created_at).toLocaleDateString()}
          </div>
        </div>
        
        {/* CONTINUE SESSION BUTTON */}
        <Button 
            size="sm" 
            onClick={handleContinue}
            className="bg-white text-black hover:bg-zinc-200 font-black uppercase text-[10px] h-10 px-4 rounded-xl shadow-xl"
        >
            <Play className="w-3 h-3 mr-2" fill="currentColor" /> Continue
        </Button>
      </header>

      <Card className="bg-[#0c0c0e] border-white/5 shadow-2xl overflow-hidden rounded-3xl ring-1 ring-white/10">
        <CardHeader className="text-center border-b border-white/5 py-4 bg-white/[0.02]">
          <CardTitle className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.3em]">Final Settlement</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <Table>
            <TableBody>
              {splitData.map((row: any, i: number) => (
                <TableRow key={i} className="border-white/5 hover:bg-white/[0.01]">
                  <TableCell className="py-6 font-bold text-white px-8">{row.name}</TableCell>
                  <TableCell className="text-right font-mono text-white text-xl px-8">{symbol}{row.amount.toFixed(2)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          
          <div className="p-6">
             <Button variant="outline" className="w-full h-12 border-white/5 text-zinc-500 hover:text-white font-bold rounded-xl uppercase tracking-widest text-[10px]" onClick={() => {
                let text = `*Bill: ${bill.bill_title} (${symbol})*\n\n`;
                splitData.forEach((r: any) => (text += `👤 *${r.name}*: ${symbol}${r.amount.toFixed(2)}\n`));
                window.open(`whatsapp://send?text=${encodeURIComponent(text)}`);
            }}>
                <Share2 className="w-4 h-4 mr-2" /> Share WhatsApp
            </Button>
          </div>
        </CardContent>
      </Card>

      <div className="p-6 border border-white/5 rounded-3xl bg-zinc-950/50">
         <p className="text-[10px] uppercase font-black text-zinc-700 mb-3 tracking-widest">System Reasoning Log</p>
         <div className="text-[10px] font-mono text-zinc-600 whitespace-pre-wrap leading-relaxed italic max-h-60 overflow-y-auto">
            {bill.reasoning_log || "No log recorded."}
         </div>
      </div>
    </div>
  );
}