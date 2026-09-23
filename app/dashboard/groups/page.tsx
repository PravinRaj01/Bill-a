'use client'
import { useState, useEffect } from "react"
import { createClient } from "@/utils/supabase/client"
import { Button } from "@/components/ui/button"
import { ChevronLeft, Trash2 } from "lucide-react"
import Link from "next/link"

export default function ManageGroups() {
  const [groups, setGroups] = useState<any[]>([])
  const supabase = createClient()

  useEffect(() => {
    fetchGroups()
  }, [])

  const fetchGroups = async () => {
    const { data: { user } } = await supabase.auth.getUser()
    if (!user) return
    const { data } = await supabase.from('saved_groups').select('*').eq('user_id', user.id).order('created_at', { ascending: false })
    if (data) setGroups(data)
  }

  const deleteGroup = async (id: string) => {
    if (!confirm("Delete this saved group?")) return
    await supabase.from('saved_groups').delete().eq('id', id)
    setGroups(groups.filter(g => g.id !== id))
  }

  return (
    <main className="p-6 md:p-10 max-w-xl mx-auto space-y-8 min-h-screen bg-black text-white">
      <Link href="/dashboard" className="flex items-center text-[10px] font-black uppercase text-zinc-500 tracking-widest hover:text-white transition-colors">
        <ChevronLeft size={14} className="mr-1" /> Back to Dashboard
      </Link>
      
      <header className="space-y-1">
        <h1 className="text-3xl font-black italic tracking-tighter uppercase">Groups</h1>
        <p className="text-zinc-500 font-mono text-[10px] uppercase tracking-widest">Manage Saved Crews</p>
      </header>

      <div className="space-y-4">
        {groups.length > 0 ? (
          groups.map(group => (
            <div key={group.id} className="bg-[#0c0c0e] border border-white/5 rounded-3xl p-6 flex justify-between items-center shadow-xl group">
              <div className="space-y-1">
                <h3 className="font-black uppercase tracking-tight text-white">{group.group_name}</h3>
                <p className="text-[10px] text-zinc-600 font-mono uppercase tracking-tight">{group.names.join(", ")}</p>
              </div>
              <Button variant="ghost" className="text-zinc-800 hover:text-red-500 hover:bg-white/5 transition-all" onClick={() => deleteGroup(group.id)}>
                <Trash2 size={18} />
              </Button>
            </div>
          ))
        ) : (
          <div className="text-center py-12 border-2 border-dashed border-zinc-900 rounded-3xl">
            <p className="text-zinc-600 font-medium italic">No saved groups found.</p>
          </div>
        )}
      </div>
    </main>
  )
}