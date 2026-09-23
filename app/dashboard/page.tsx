'use client'

import { useState, useEffect } from "react"
import Link from "next/link"
import { createClient } from "@/utils/supabase/client"
import { Button } from "@/components/ui/button"
import { Plus, Users, ChevronRight, Zap, Camera, Share2 } from "lucide-react"
import { Card } from "@/components/ui/card"

export default function DashboardPage() {
  const [groups, setGroups] = useState<any[]>([])
  const supabase = createClient()

  useEffect(() => {
    const fetchGroups = async () => {
      const { data: { user } } = await supabase.auth.getUser()
      if (user) {
        const { data } = await supabase
          .from('saved_groups')
          .select('*')
          .eq('user_id', user.id)
          .order('created_at', { ascending: false })
        if (data) setGroups(data)
      }
    }
    fetchGroups()
  }, [])

  return (
    <div className="min-h-screen bg-black text-white p-6 pb-24 md:p-10 max-w-6xl mx-auto animate-in fade-in">

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 lg:gap-12 items-start">
        
        {/* LEFT COLUMN: Actions & Instructions */}
        <div className="space-y-8">
          {/* New Session Button */}
          <Link href="/dashboard/new" className="block group">
            <Button className="w-full h-64 md:h-80 bg-white text-black hover:bg-zinc-200 rounded-[2.5rem] flex flex-col items-center justify-center gap-6 transition-all active:scale-95 shadow-2xl border-0">
              <Plus size={72} strokeWidth={3} className="group-hover:scale-110 transition-transform duration-300" />
              <span className="text-3xl font-black uppercase tracking-tighter">New Session</span>
            </Button>
          </Link>

          {/* Instructions */}
          <section className="bg-zinc-900/30 border border-white/5 rounded-3xl p-8 space-y-6">
            <h3 className="text-[10px] font-black uppercase tracking-[0.2em] text-zinc-500">How it works</h3>
            <div className="space-y-5">
              <div className="flex gap-4 items-start">
                <div className="w-10 h-10 rounded-full bg-zinc-800 flex items-center justify-center shrink-0">
                  <Zap size={16} className="text-white" />
                </div>
                <div>
                  <h4 className="text-sm font-bold text-white">Setup</h4>
                  <p className="text-xs text-zinc-400 font-medium leading-relaxed mt-1">Create a group or pick a saved crew.</p>
                </div>
              </div>
              <div className="flex gap-4 items-start">
                <div className="w-10 h-10 rounded-full bg-zinc-800 flex items-center justify-center shrink-0">
                  <Camera size={16} className="text-white" />
                </div>
                 <div>
                  <h4 className="text-sm font-bold text-white">Scan</h4>
                  <p className="text-xs text-zinc-400 font-medium leading-relaxed mt-1">Snap a receipt. AI extracts the items.</p>
                </div>
              </div>
              <div className="flex gap-4 items-start">
                <div className="w-10 h-10 rounded-full bg-zinc-800 flex items-center justify-center shrink-0">
                  <Share2 size={16} className="text-white" />
                </div>
                 <div>
                  <h4 className="text-sm font-bold text-white">Share</h4>
                  <p className="text-xs text-zinc-400 font-medium leading-relaxed mt-1">Split costs and share via WhatsApp.</p>
                </div>
              </div>
            </div>
          </section>
        </div>

        {/* RIGHT COLUMN: Saved Groups */}
        <div className="space-y-4 h-full">
          <div className="flex justify-between items-end px-2 mb-2">
            <div className="flex items-center gap-2 text-zinc-500">
              <Users size={16} />
              <h2 className="text-xs font-black uppercase tracking-widest">Saved Groups</h2>
            </div>
            <Link href="/dashboard/groups" className="text-[10px] font-bold text-zinc-600 hover:text-white transition-colors uppercase tracking-widest">
              Manage
            </Link>
          </div>

          <div className="space-y-4">
            {groups.length > 0 ? (
              groups.map((group) => (
                // Added w-full here to ensure the link fills the grid column
                <Link key={group.id} href={`/dashboard/new?group_id=${group.id}`} className="block w-full">
                  <Card className="bg-[#0c0c0e] border border-white/5 hover:border-white/20 transition-all rounded-3xl p-6 group flex items-center justify-between h-24">
                    
                    {/* FIXED: Added flex-1 and min-w-0 to fix disappearance */}
                    <div className="flex flex-col justify-center gap-1.5 flex-1 min-w-0 mr-4">
                      <h3 className="font-bold text-white uppercase tracking-tight truncate text-base leading-none">
                        {group.group_name}
                      </h3>
                      <p className="text-[11px] text-zinc-500 font-mono uppercase truncate leading-none">
                        {group.names.join(", ")}
                      </p>
                    </div>

                    <div className="w-12 h-12 bg-zinc-900 rounded-full flex items-center justify-center shrink-0 group-hover:bg-white group-hover:text-black transition-colors">
                      <ChevronRight size={20} />
                    </div>
                  </Card>
                </Link>
              ))
            ) : (
              <div className="h-64 flex flex-col items-center justify-center text-center border-2 border-dashed border-white/5 rounded-[2.5rem]">
                <Users className="w-10 h-10 text-zinc-800 mb-4" />
                <p className="text-xs text-zinc-600 uppercase tracking-widest font-bold">No saved groups</p>
                <p className="text-[10px] text-zinc-700 mt-2">Save your frequent squads here.</p>
              </div>
            )}
          </div>
        </div>

      </div>
    </div>
  )
}