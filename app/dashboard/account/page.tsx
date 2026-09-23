"use client";

import { useEffect, useState } from "react";
import { createClient } from "@/utils/supabase/client";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { 
  LogOut, 
  User, 
  Wallet, 
  History, 
  TrendingUp,
  Mail
} from "lucide-react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

export default function AccountPage() {
  const [user, setUser] = useState<any>(null);
  const [stats, setStats] = useState({ totalBills: 0, totalAmount: 0 });
  const [loading, setLoading] = useState(true);
  const supabase = createClient();
  const router = useRouter();

  useEffect(() => {
    const getData = async () => {
      const { data: { user } } = await supabase.auth.getUser();
      
      if (user) {
        setUser(user);
        
        // Fetch stats from bill_history
        const { data: history } = await supabase
          .from('bill_history')
          .select('total_amount')
          .eq('user_id', user.id);
          
        if (history) {
          const total = history.reduce((sum, item) => sum + (item.total_amount || 0), 0);
          setStats({
            totalBills: history.length,
            totalAmount: total
          });
        }
      }
      setLoading(false);
    };
    getData();
  }, []);

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    router.replace("/");
  };

  return (
    <div className="min-h-screen bg-black text-white p-6 md:p-10 max-w-xl mx-auto animate-in fade-in duration-500">
      
      {/* Header */}
      <header className="space-y-1 mb-8">
        <h1 className="text-3xl font-black tracking-tighter uppercase italic text-white">My Account</h1>
      </header>

      <div className="space-y-6">
        
        {/* Profile Card */}
        <Card className="bg-[#0c0c0e] border border-white/5 rounded-3xl overflow-hidden">
          <CardContent className="p-8 flex flex-col items-center text-center space-y-4">
            <div className="w-24 h-24 rounded-full bg-gradient-to-tr from-zinc-800 to-zinc-900 p-1 ring-1 ring-white/10">
              <Avatar className="w-full h-full rounded-full">
                <AvatarImage src={user?.user_metadata?.avatar_url} />
                <AvatarFallback className="bg-black text-zinc-500 font-bold text-2xl">
                  {user?.email?.charAt(0).toUpperCase()}
                </AvatarFallback>
              </Avatar>
            </div>
            
            <div className="space-y-1">
              <h2 className="text-lg font-bold text-white tracking-tight">
                {user?.user_metadata?.full_name || "Bill.a User"}
              </h2>
              <div className="flex items-center justify-center gap-2 text-xs text-zinc-500 font-mono bg-white/5 py-1 px-3 rounded-full w-fit mx-auto">
                <Mail size={12} />
                {user?.email}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 gap-4">
          <Card className="bg-[#0c0c0e] border border-white/5 rounded-3xl p-6 relative overflow-hidden group hover:border-white/20 transition-all">
            <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
              <History size={48} />
            </div>
            <div className="space-y-1 relative z-10">
              <span className="text-[10px] font-black uppercase tracking-widest text-zinc-500">Sessions</span>
              <div className="text-3xl font-black text-white tracking-tighter">
                {stats.totalBills}
              </div>
            </div>
          </Card>

          <Card className="bg-[#0c0c0e] border border-white/5 rounded-3xl p-6 relative overflow-hidden group hover:border-white/20 transition-all">
            <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
              <Wallet size={48} />
            </div>
            <div className="space-y-1 relative z-10">
              <span className="text-[10px] font-black uppercase tracking-widest text-zinc-500">Total Value</span>
              <div className="text-3xl font-black text-white tracking-tighter">
                <span className="text-base align-top opacity-50 mr-0.5">$</span>
                {stats.totalAmount.toLocaleString(undefined, { maximumFractionDigits: 0 })}
              </div>
            </div>
          </Card>
        </div>

        {/* Action Buttons */}
        <div className="pt-4 space-y-3">
           <Button 
            onClick={handleSignOut}
            className="w-full h-14 bg-zinc-900 hover:bg-zinc-800 text-white border border-white/5 rounded-2xl font-bold uppercase tracking-widest text-xs flex items-center justify-between px-6 group"
          >
            <span>Sign Out</span>
            <LogOut size={16} className="text-zinc-500 group-hover:text-white transition-colors" />
          </Button>

          <div className="text-center pt-4">
             <p className="text-[10px] text-zinc-700 font-mono uppercase">User ID: {user?.id?.slice(0, 8)}...</p>
          </div>
        </div>

      </div>
    </div>
  );
}