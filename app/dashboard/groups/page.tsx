'use client'
import { useState, useEffect } from "react"
import { deleteGroup as deleteGroupAction, listGroups } from "@/lib/actions/groups"
import { getCurrentUser } from "@/lib/actions/user"
import type { SavedGroup } from "@/lib/db/schema"
import { Button } from "@/components/ui/button"
import { ChevronLeft, Trash2 } from "lucide-react"
import Link from "next/link"

export default function ManageGroups() {
  const [groups, setGroups] = useState<SavedGroup[]>([])
  // null = still loading, false = guest, true = signed in
  const [signedIn, setSignedIn] = useState<boolean | null>(null)

  useEffect(() => {
    getCurrentUser().then(async (user) => {
      setSignedIn(!!user)
      if (user) setGroups(await listGroups())
    })
  }, [])

  const deleteGroup = async (id: string) => {
    if (!confirm("Delete this saved group?")) return
    try {
      // Server-side ownership check: only the signed-in user's own group can be deleted.
      await deleteGroupAction(id)
      setGroups(groups.filter(g => g.id !== id))
    } catch {
      alert("Couldn't delete that group. Please try again.")
    }
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
                <h3 className="font-black uppercase tracking-tight text-white">{group.groupName}</h3>
                <p className="text-[10px] text-zinc-600 font-mono uppercase tracking-tight">{group.names.join(", ")}</p>
              </div>
              <Button variant="ghost" className="text-zinc-800 hover:text-red-500 hover:bg-white/5 transition-all" onClick={() => deleteGroup(group.id)}>
                <Trash2 size={18} />
              </Button>
            </div>
          ))
        ) : (
          <div className="text-center py-12 border-2 border-dashed border-zinc-900 rounded-3xl">
            {signedIn === false ? (
              <p className="text-zinc-600 font-medium italic">
                <Link href="/" className="underline underline-offset-4 hover:text-white">Log in</Link> to save and reuse groups.
              </p>
            ) : (
              <p className="text-zinc-600 font-medium italic">{signedIn === null ? "Loading..." : "No saved groups found."}</p>
            )}
          </div>
        )}
      </div>
    </main>
  )
}