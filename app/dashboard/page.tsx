'use client'

import { useState, useEffect } from "react"
import Link from "next/link"
import { listGroups } from "@/lib/actions/groups"
import type { SavedGroup } from "@/lib/db/schema"
import { Users, ChevronRight, Zap, Camera, Share2, ArrowUpRight, Calculator } from "lucide-react"
import { Card } from "@/components/ui/card"
import { MeshHero } from "@/components/ui/mesh-hero"

// The shader colours: graphite into a warm coral glow (the same coral as the primary buttons).
const SCAN_COLORS = ["#0e0f12", "#3a1b22", "#ff6b57", "#1f2228"]

export default function DashboardPage() {
  const [groups, setGroups] = useState<SavedGroup[]>([])

  useEffect(() => {
    // Resolved on the server from the session: guests simply get an empty list.
    listGroups().then(setGroups).catch(() => setGroups([]))
  }, [])

  return (
    <div className="min-h-screen bg-background text-white p-6 pb-24 md:p-10 max-w-6xl mx-auto animate-in fade-in space-y-10">

      {/* Two ways in: a receipt, or just an amount. */}
      <section aria-label="Start a split" className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <Link href="/dashboard/new" data-testid="tile-scan" className="group block rounded-[2rem] outline-none focus-visible:ring-4 focus-visible:ring-white/20">
          <MeshHero colors={SCAN_COLORS} className="h-44 rounded-[2rem] border border-white/10 transition-transform duration-200 group-active:scale-[0.98] sm:h-56">
            <div className="flex h-44 flex-col justify-between p-6 sm:h-56">
              <div className="flex items-start justify-between">
                <span className="flex size-12 items-center justify-center rounded-full bg-white/15 backdrop-blur">
                  <Camera className="size-6" />
                </span>
                <ArrowUpRight className="size-6 opacity-60 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
              </div>
              <div>
                <h2 className="text-2xl font-bold tracking-tight">Scan receipt</h2>
                <p className="mt-1 text-sm text-white/70">Snap it, we read the items.</p>
              </div>
            </div>
          </MeshHero>
        </Link>

        <Link href="/dashboard/new?mode=quick" data-testid="tile-quick" className="group block rounded-[2rem] outline-none focus-visible:ring-4 focus-visible:ring-white/20">
          <div className="flex h-44 flex-col justify-between rounded-[2rem] border border-white/10 bg-surface p-6 transition-all duration-200 group-hover:border-white/25 group-active:scale-[0.98] sm:h-56">
            <div className="flex items-start justify-between">
              <span className="flex size-12 items-center justify-center rounded-full bg-coral/15 text-coral">
                <Calculator className="size-6" />
              </span>
              <ArrowUpRight className="size-6 opacity-60 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
            </div>
            <div>
              <h2 className="text-2xl font-bold tracking-tight">Quick split</h2>
              <p className="mt-1 text-sm text-muted-foreground">Just an amount. No receipt needed.</p>
            </div>
          </div>
        </Link>
      </section>

      <div className="grid grid-cols-1 gap-10 lg:grid-cols-2 lg:items-start lg:gap-12">
        {/* Instructions */}
        <section className="bg-surface border border-white/5 rounded-3xl p-8 space-y-6">
          <h3 className="text-xs font-semibold tracking-widest uppercase text-muted-foreground">How it works</h3>
          <div className="space-y-5">
            <div className="flex gap-4 items-start">
              <div className="w-10 h-10 rounded-full bg-surface-2 flex items-center justify-center shrink-0">
                <Zap size={16} className="text-white" />
              </div>
              <div>
                <h4 className="text-sm font-bold text-white">Setup</h4>
                <p className="text-sm text-muted-foreground leading-relaxed mt-1">Create a group or pick a saved crew.</p>
              </div>
            </div>
            <div className="flex gap-4 items-start">
              <div className="w-10 h-10 rounded-full bg-surface-2 flex items-center justify-center shrink-0">
                <Camera size={16} className="text-white" />
              </div>
              <div>
                <h4 className="text-sm font-bold text-white">Scan or enter</h4>
                <p className="text-sm text-muted-foreground leading-relaxed mt-1">Snap a receipt, or just type an amount.</p>
              </div>
            </div>
            <div className="flex gap-4 items-start">
              <div className="w-10 h-10 rounded-full bg-surface-2 flex items-center justify-center shrink-0">
                <Share2 size={16} className="text-white" />
              </div>
              <div>
                <h4 className="text-sm font-bold text-white">Split and share</h4>
                <p className="text-sm text-muted-foreground leading-relaxed mt-1">Equal, by sliders or in words, then share via WhatsApp.</p>
              </div>
            </div>
          </div>
        </section>

        {/* Saved groups */}
        <div className="space-y-4">
          <div className="flex justify-between items-end px-2 mb-2">
            <div className="flex items-center gap-2 text-muted-foreground">
              <Users size={16} />
              <h2 className="text-xs font-semibold tracking-widest uppercase">Saved groups</h2>
            </div>
            <Link href="/dashboard/groups" className="text-xs font-semibold text-muted-foreground hover:text-white transition-colors uppercase tracking-widest">
              Manage
            </Link>
          </div>

          <div className="space-y-4">
            {groups.length > 0 ? (
              groups.map((group) => (
                <Link key={group.id} href={`/dashboard/new?group_id=${group.id}`} className="block w-full">
                  <Card className="bg-surface border border-white/5 hover:border-white/20 transition-all rounded-3xl p-6 group flex items-center justify-between h-24">
                    <div className="flex flex-col justify-center gap-1.5 flex-1 min-w-0 mr-4">
                      <h3 className="font-bold text-white truncate text-base leading-none">
                        {group.groupName}
                      </h3>
                      <p className="text-sm text-muted-foreground truncate leading-none">
                        {group.names.join(", ")}
                      </p>
                    </div>

                    <div className="w-12 h-12 bg-surface-2 rounded-full flex items-center justify-center shrink-0 group-hover:bg-white group-hover:text-black transition-colors">
                      <ChevronRight size={20} />
                    </div>
                  </Card>
                </Link>
              ))
            ) : (
              <div className="h-48 flex flex-col items-center justify-center text-center border-2 border-dashed border-white/5 rounded-[2rem]">
                <Users className="w-10 h-10 text-zinc-700 mb-4" />
                <p className="text-sm text-muted-foreground font-semibold">No saved groups</p>
                <p className="text-xs text-zinc-600 mt-2">Save your frequent squads here.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
