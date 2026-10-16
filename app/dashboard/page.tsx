'use client'

import { useState, useEffect } from "react"
import Link from "next/link"
import { ArrowUpRight, Calculator, Camera, ChevronRight, LogIn, Users } from "lucide-react"

import { AvatarStack } from "@/components/person-avatar"
import { PageTransition } from "@/components/page-transition"
import { MeshHero } from "@/components/ui/mesh-hero"
import { listGroups } from "@/lib/actions/groups"
import { getBalances, listBills, type BillListEntry } from "@/lib/actions/history"
import { getCurrentUser } from "@/lib/actions/user"
import type { SavedGroup } from "@/lib/db/schema"
import { formatMoney } from "@/lib/money"
import { localDate, shortDate } from "@/lib/split/settle"
import { settlePill } from "@/lib/ui/settle-pill"
import { cn } from "@/lib/utils"

// The shader colours: graphite into a warm coral glow (the same coral as the primary buttons).
const SCAN_COLORS = ["#0e0f12", "#3a1b22", "#ff6b57", "#1f2228"]

function greeting(hour: number | null): string {
  if (hour === null) return "Hello"
  if (hour < 5) return "Up late"
  if (hour < 12) return "Good morning"
  if (hour < 18) return "Good afternoon"
  return "Good evening"
}

export default function DashboardPage() {
  // null = still loading, false = guest, true = signed in
  const [signedIn, setSignedIn] = useState<boolean | null>(null)
  const [firstName, setFirstName] = useState<string | null>(null)
  const [groups, setGroups] = useState<SavedGroup[]>([])
  const [latest, setLatest] = useState<BillListEntry | null>(null)
  // What people owe you / you owe, per currency, from tracked bills (empty for guests and when all square).
  const [owed, setOwed] = useState<[string, { owedToYou: number; youOwe: number }][]>([])
  // The date and hour are the user's own, so they wait for the browser.
  const [now, setNow] = useState<{ hour: number; today: string } | null>(null)

  useEffect(() => {
    // After mount, so the server-rendered page (which can't know the user's clock) never disagrees with it.
    queueMicrotask(() => setNow({ hour: new Date().getHours(), today: localDate() }))
    getCurrentUser()
      .then((u) => {
        setSignedIn(!!u)
        setFirstName(u?.name?.trim().split(/\s+/)[0] || null)
      })
      .catch(() => setSignedIn(false))
    // Resolved on the server from the session: guests simply get empty results.
    listGroups().then(setGroups).catch(() => setGroups([]))
    listBills().then((b) => setLatest(b[0] ?? null)).catch(() => setLatest(null))
    getBalances()
      .then((r) => {
        const by = new Map<string, { owedToYou: number; youOwe: number }>()
        for (const p of r.people) {
          const t = by.get(p.currency) ?? { owedToYou: 0, youOwe: 0 }
          if (p.net > 0) t.owedToYou += p.net
          else t.youOwe += -p.net
          by.set(p.currency, t)
        }
        setOwed([...by])
      })
      .catch(() => setOwed([]))
  }, [])

  const pill = latest && now ? settlePill(latest, now.today) : null

  return (
    <PageTransition>
      <div className="mx-auto min-h-screen max-w-6xl space-y-8 bg-background p-6 pb-24 text-white md:p-10">
        <header className="space-y-1">
          <h1 data-testid="greeting" className="text-3xl font-black tracking-tight">
            {greeting(now?.hour ?? null)}
            {firstName ? `, ${firstName}` : ""}
          </h1>
          <p className="text-sm text-muted-foreground">{signedIn === false ? "Split a bill in under a minute." : "Who are we splitting with today?"}</p>
        </header>

        {/* Two ways in: a receipt, or just an amount. */}
        <section aria-label="Start a split" className="grid grid-cols-2 gap-3 sm:gap-4">
          <Link href="/dashboard/new" transitionTypes={["nav-forward"]} data-testid="tile-scan" className="group block rounded-[2rem] outline-none focus-visible:ring-4 focus-visible:ring-white/20">
            <MeshHero colors={SCAN_COLORS} className="h-40 rounded-[2rem] border border-white/10 transition-transform duration-200 group-active:scale-[0.98] sm:h-52">
              <div className="flex h-40 flex-col justify-between p-4 sm:h-52 sm:p-6">
                <div className="flex items-start justify-between">
                  <span className="flex size-11 items-center justify-center rounded-full bg-white/15 backdrop-blur">
                    <Camera className="size-5" />
                  </span>
                  <ArrowUpRight className="size-5 opacity-60 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
                </div>
                <div>
                  <h2 className="text-lg font-bold tracking-tight sm:text-2xl">Scan receipt</h2>
                  <p className="mt-0.5 text-xs text-white/70 sm:text-sm">We read the items.</p>
                </div>
              </div>
            </MeshHero>
          </Link>

          <Link href="/dashboard/new?mode=quick" transitionTypes={["nav-forward"]} data-testid="tile-quick" className="group block rounded-[2rem] outline-none focus-visible:ring-4 focus-visible:ring-white/20">
            <div className="flex h-40 flex-col justify-between rounded-[2rem] border border-white/10 bg-surface p-4 transition-all duration-200 group-hover:border-white/25 group-active:scale-[0.98] sm:h-52 sm:p-6">
              <div className="flex items-start justify-between">
                <span className="flex size-11 items-center justify-center rounded-full bg-coral/15 text-coral">
                  <Calculator className="size-5" />
                </span>
                <ArrowUpRight className="size-5 opacity-60 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
              </div>
              <div>
                <h2 className="text-lg font-bold tracking-tight sm:text-2xl">Quick split</h2>
                <p className="mt-0.5 text-xs text-muted-foreground sm:text-sm">Just an amount.</p>
              </div>
            </div>
          </Link>
        </section>

        {owed.length > 0 && (
          <Link href="/dashboard/balances" transitionTypes={["nav-forward"]} data-testid="home-owed" className="flex min-h-16 items-center justify-between gap-4 rounded-3xl border border-white/5 bg-surface p-5 transition-colors hover:border-white/20">
            <div className="space-y-0.5 text-sm text-muted-foreground">
              {owed.map(([cur, t]) => (
                <div key={cur}>
                  {t.owedToYou > 0 && <p>You&apos;re owed <strong className="text-lg font-bold text-success tabular-nums">{formatMoney(t.owedToYou, cur)}</strong></p>}
                  {t.youOwe > 0 && <p>You owe <strong className="text-lg font-bold text-coral tabular-nums">{formatMoney(t.youOwe, cur)}</strong></p>}
                </div>
              ))}
            </div>
            <span className="flex items-center gap-1 text-sm font-semibold text-muted-foreground">Balances <ChevronRight size={16} /></span>
          </Link>
        )}

        {signedIn === false && (
          <section data-testid="home-guest" className="flex flex-col gap-4 rounded-3xl border border-white/5 bg-surface p-6 sm:flex-row sm:items-center sm:justify-between">
            <div className="space-y-1">
              <h2 className="text-base font-bold">Keep your splits</h2>
              <p className="text-sm text-muted-foreground">Log in to save your history, see who owes you, and reuse your groups.</p>
            </div>
            <Link href="/" className="flex min-h-12 shrink-0 items-center justify-center gap-2 rounded-xl bg-coral px-6 font-bold text-coral-foreground">
              <LogIn className="size-4" /> Log in
            </Link>
          </section>
        )}

        {latest && (
          <section aria-label="Pick up where you left off" className="space-y-3">
            <h2 className="px-1 text-xs font-semibold tracking-widest text-muted-foreground uppercase">Pick up where you left off</h2>
            <Link
              href={`/dashboard/history/${latest.id}`}
              transitionTypes={["nav-forward"]}
              data-testid="home-latest"
              className="block space-y-3 rounded-3xl border border-white/5 bg-surface p-5 transition-colors hover:border-white/20"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <h3 className="truncate text-lg font-bold">{latest.billTitle}</h3>
                  <p className="font-mono text-xs tracking-wider text-muted-foreground uppercase">{shortDate(latest.createdAt)}</p>
                </div>
                <p className="shrink-0 text-xl font-black tabular-nums">{formatMoney(latest.totalAmount, latest.currency || "RM")}</p>
              </div>
              <div className="flex items-center justify-between gap-3">
                {latest.people.length > 0 ? <AvatarStack names={latest.people} max={4} /> : <span />}
                <span className="flex items-center gap-1 text-xs font-semibold text-muted-foreground">
                  Open <ChevronRight size={14} />
                </span>
              </div>
              {pill && (
                <span data-testid="home-latest-pill" className={cn("inline-block rounded-full border px-3 py-1 text-xs font-semibold", pill.tone)}>
                  {pill.text}
                </span>
              )}
            </Link>
          </section>
        )}

        {/* Saved groups */}
        {signedIn !== false && (
          <section aria-label="Saved groups" className="space-y-3">
            <div className="flex items-center justify-between px-1">
              <h2 className="flex items-center gap-2 text-xs font-semibold tracking-widest text-muted-foreground uppercase">
                <Users size={14} /> Your groups
              </h2>
              <Link href="/dashboard/groups" transitionTypes={["nav-forward"]} className="flex min-h-11 items-center text-xs font-semibold tracking-widest text-muted-foreground uppercase transition-colors hover:text-white">
                Manage
              </Link>
            </div>

            {groups.length > 0 ? (
              <ul className="-mx-6 flex snap-x snap-mandatory gap-3 overflow-x-auto px-6 pb-2 md:mx-0 md:grid md:grid-cols-2 md:overflow-visible md:px-0 lg:grid-cols-3">
                {groups.map((group) => (
                  <li key={group.id} className="w-64 shrink-0 snap-start md:w-auto">
                    <Link
                      href={`/dashboard/new?group_id=${group.id}`}
                      transitionTypes={["nav-forward"]}
                      data-testid="home-group"
                      className="flex h-full flex-col justify-between gap-4 rounded-3xl border border-white/5 bg-surface p-5 transition-colors hover:border-white/20"
                    >
                      <AvatarStack names={group.names} max={4} />
                      <div className="min-w-0">
                        <h3 className="truncate text-base font-bold">{group.groupName}</h3>
                        <p className="truncate text-sm text-muted-foreground">{group.names.join(", ")}</p>
                      </div>
                      <span className="flex min-h-11 items-center justify-center rounded-xl bg-surface-2 text-sm font-semibold text-zinc-200">Split</span>
                    </Link>
                  </li>
                ))}
              </ul>
            ) : (
              signedIn && (
                <div data-testid="home-no-groups" className="space-y-1 rounded-3xl border-2 border-dashed border-white/5 px-6 py-8 text-center">
                  <p className="text-sm font-semibold text-muted-foreground">No saved groups yet</p>
                  <p className="text-xs text-muted-foreground/70">On your next split, choose &ldquo;Save this group&rdquo; and your crew shows up here.</p>
                </div>
              )
            )}
          </section>
        )}
      </div>
    </PageTransition>
  )
}
