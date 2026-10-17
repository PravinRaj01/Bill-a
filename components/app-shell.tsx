'use client'

import { useState } from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { m } from "motion/react"
import { Bell, Home, History, Scale, User, UserRound, LogOut, LogIn, Plus } from "lucide-react"
import { cn } from "@/lib/utils"
import { Badge } from "@/components/ui/badge"
import { StartSheet } from "@/components/nav/StartSheet"
import { NotificationBell, UnreadBadge, useUnreadCount } from "@/components/notifications/NotificationBell"
import { PersonAvatar } from "@/components/person-avatar"
import { useProfile } from "@/components/profile-provider"
import { signOutAction } from "@/app/auth/actions"
import { Button } from "@/components/ui/button";

type NavItem = { name: string; href: string; icon: typeof Home }

export function AppShell({ children, signedIn }: { children: React.ReactNode; signedIn: boolean }) {
  const pathname = usePathname()
  const profile = useProfile()
  const [startOpen, setStartOpen] = useState(false)
  const unread = useUnreadCount(signedIn)

  // History, Balances and Account require a login (see lib/auth.config.ts), so guests
  // don't get links that would just bounce them to the login page.
  const home: NavItem = { name: 'Home', href: '/dashboard', icon: Home }
  const left: NavItem[] = signedIn ? [home, { name: 'History', href: '/dashboard/history', icon: History }] : [home]
  const right: NavItem[] = signedIn
    ? [
        { name: 'Balances', href: '/dashboard/balances', icon: Scale },
        { name: 'Account', href: '/dashboard/account', icon: User },
      ]
    : []
  const navItems = [...left, ...right]

  // /dashboard/history/abc keeps History lit; Home only lights on the exact page.
  const isActive = (href: string) => (href === '/dashboard' ? pathname === href : pathname === href || pathname.startsWith(href + '/'))

  const handleLogout = () => {
    // Wipe the offline copies of the app pages before signing out, so nothing tied to this
    // session is left behind on a shared device.
    navigator.serviceWorker?.controller?.postMessage({ type: "clear-pages" })
    return signOutAction()
  }

  const tab = (item: NavItem) => {
    const active = isActive(item.href)
    return (
      <Link
        key={item.name}
        href={item.href}
        prefetch={true} // FOR SPEED
        aria-current={active ? "page" : undefined}
        className={cn(
          "relative flex min-h-14 flex-col items-center justify-center gap-0.5 rounded-2xl outline-none transition-colors focus-visible:ring-2 focus-visible:ring-white/30",
          active ? "text-foreground" : "text-muted-foreground hover:text-foreground"
        )}
      >
        {active && <m.span layoutId="nav-pill" className="absolute inset-x-2 inset-y-1 -z-10 rounded-2xl bg-white/10" transition={{ type: "spring", stiffness: 500, damping: 40 }} />}
        <item.icon className="size-5" />
        <span className="text-[10px] font-bold tracking-tight">{item.name}</span>
      </Link>
    )
  }

  return (
    <div className="flex min-h-screen flex-col text-foreground">
      <header style={{ viewTransitionName: "app-header" }} className="flex h-14 shrink-0 items-center justify-between px-6 border-b border-white/8 bg-background/50 backdrop-blur-xl sticky top-0 z-50">
        <div className="flex items-center gap-2">
          <img src="/logo.png" alt="" className="size-6 rounded-md bg-white object-contain" />
          <span className="text-sm font-bold tracking-tighter italic">Bill.a</span>
        </div>
        {signedIn ? (
          <div className="flex items-center gap-1">
            <NotificationBell count={unread} />
            {/* Your own portrait, straight to your profile. */}
            <Link href="/dashboard/account?tab=profile" aria-label="Your profile" data-testid="header-avatar" className="flex size-11 items-center justify-center rounded-full outline-none focus-visible:ring-2 focus-visible:ring-white/30">
              <PersonAvatar name={profile.meName} compact className="size-9" />
            </Link>
          </div>
        ) : (
          <Badge variant="outline" className="text-[10px] border-white/10 opacity-50 uppercase tracking-widest">
            V1.0
          </Badge>
        )}
      </header>

      <div className="flex flex-1">
        {/* Desktop sidebar */}
        <aside className="hidden md:flex w-64 flex-col border-r border-white/8 bg-white/[0.035] p-6 backdrop-blur-xl">
          <Button type="button" size="lg" data-testid="new-split" onClick={() => setStartOpen(true)} className="mb-6 w-full">
            <Plus className="size-5" strokeWidth={3} /> New split
          </Button>
          <nav className="flex-1 space-y-2">
            {navItems.map((item) => (
              <Link
                key={item.name}
                href={item.href}
                prefetch={true} // FOR SPEED
                aria-current={isActive(item.href) ? "page" : undefined}
                className={cn(
                  "flex items-center gap-3 px-3 py-2 rounded-xl transition-all",
                  isActive(item.href) ? "bg-primary/15 text-primary font-bold" : "text-muted-foreground hover:bg-white/5 hover:text-foreground"
                )}
              >
                <item.icon className="w-5 h-5" />
                {item.name}
              </Link>
            ))}
          </nav>
          {signedIn && (
            <Link
              href="/dashboard/notifications"
              data-testid="nav-notifications"
              aria-current={isActive('/dashboard/notifications') ? "page" : undefined}
              className={cn(
                "mb-2 flex items-center gap-3 px-3 py-2 rounded-xl transition-all",
                isActive('/dashboard/notifications') ? "bg-primary/15 text-primary font-bold" : "text-muted-foreground hover:bg-white/5 hover:text-foreground"
              )}
            >
              <Bell className="w-5 h-5" />
              Notifications
              <UnreadBadge count={unread} className="ml-auto" />
            </Link>
          )}
          {signedIn && (
            <Link
              href="/dashboard/people"
              data-testid="nav-people"
              aria-current={isActive('/dashboard/people') ? "page" : undefined}
              className={cn(
                "mb-2 flex items-center gap-3 px-3 py-2 rounded-xl transition-all",
                isActive('/dashboard/people') ? "bg-primary/15 text-primary font-bold" : "text-muted-foreground hover:bg-white/5 hover:text-foreground"
              )}
            >
              <UserRound className="w-5 h-5" />
              People
            </Link>
          )}
          {signedIn ? (
            <button onClick={handleLogout} className="flex items-center gap-3 px-3 py-2 mt-auto text-muted-foreground hover:text-destructive">
              <LogOut className="w-5 h-5" /> Logout
            </button>
          ) : (
            <Link href="/" className="flex items-center gap-3 px-3 py-2 mt-auto text-muted-foreground hover:text-foreground">
              <LogIn className="w-5 h-5" /> Log in
            </Link>
          )}
        </aside>

        <main className="min-w-0 flex-1 pb-24 md:pb-0">
          {children}
        </main>
      </div>

      {/* Phone bottom nav: two tabs, the raised + in the middle, two tabs. Guests get Home, +, Log in. */}
      <nav
        aria-label="Main"
        className="md:hidden fixed inset-x-0 bottom-0 z-50 border-t border-white/10 bg-background/55 px-3 pt-2 pb-[max(0.5rem,env(safe-area-inset-bottom))] backdrop-blur-2xl"
      >
        <div className={cn("grid items-center", signedIn ? "grid-cols-5" : "grid-cols-3")}>
          {left.map(tab)}
          <div className="flex justify-center">
            <button
              type="button"
              data-testid="nav-plus"
              aria-label="Start a split"
              onClick={() => setStartOpen(true)}
              className="-mt-7 flex size-14 items-center justify-center rounded-full bg-coral text-coral-foreground shadow-[var(--coral-glow)] outline-none ring-4 ring-background transition-transform focus-visible:ring-white/40 active:scale-95"
            >
              <Plus className="size-7" strokeWidth={3} />
            </button>
          </div>
          {signedIn ? (
            right.map(tab)
          ) : (
            <Link href="/" className="flex min-h-14 flex-col items-center justify-center gap-0.5 rounded-2xl text-muted-foreground hover:text-foreground">
              <LogIn className="size-5" />
              <span className="text-[10px] font-bold tracking-tight">Log in</span>
            </Link>
          )}
        </div>
      </nav>

      <StartSheet open={startOpen} onOpenChange={setStartOpen} signedIn={signedIn} />
    </div>
  )
}
