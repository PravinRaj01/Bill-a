'use client'

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Home, History, User, LogOut } from "lucide-react"
import { cn } from "@/lib/utils"
import { Badge } from "@/components/ui/badge"
import { createClient } from "@/utils/supabase/client"
import { useRouter } from "next/navigation"

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()
  const router = useRouter()
  const supabase = createClient()

  const navItems = [
    { name: 'Home', href: '/dashboard', icon: Home },
    { name: 'History', href: '/dashboard/history', icon: History },
    { name: 'Account', href: '/dashboard/account', icon: User },
  ]

  const handleLogout = async () => {
    await supabase.auth.signOut()
    router.push('/')
    router.refresh()
  }

  return (
    <div className="flex min-h-screen bg-black text-white flex-col">
      {/* RESTORED HEADER */}
      <header className="flex h-14 shrink-0 items-center justify-between px-6 border-b border-white/5 bg-zinc-950/50 backdrop-blur-md sticky top-0 z-50">
        <div className="flex items-center gap-2">
          <div className="bg-white p-0.5 rounded shadow-sm">
            <img src="/icon.png" alt="logo" className="w-4 h-4 object-contain" />
          </div>
          <span className="text-sm font-bold tracking-tighter italic">Bill.a</span>
        </div>
        <Badge variant="outline" className="text-[10px] border-white/10 opacity-50 uppercase tracking-widest">
          V1.0
        </Badge>
      </header>

      <div className="flex flex-1">
        {/* Desktop Sidebar (Stayed for Desktop users) */}
        <aside className="hidden md:flex w-64 flex-col border-r border-white/5 bg-zinc-950 p-6">
          <nav className="flex-1 space-y-2">
            {navItems.map((item) => (
              <Link
                key={item.name}
                href={item.href}
                prefetch={true} // FOR SPEED
                className={cn(
                  "flex items-center gap-3 px-3 py-2 rounded-xl transition-all",
                  pathname === item.href ? "bg-white text-black font-bold" : "text-zinc-400 hover:text-white"
                )}
              >
                <item.icon className="w-5 h-5" />
                {item.name}
              </Link>
            ))}
          </nav>
          <button onClick={handleLogout} className="flex items-center gap-3 px-3 py-2 mt-auto text-zinc-500 hover:text-red-400">
            <LogOut className="w-5 h-5" /> Logout
          </button>
        </aside>

        {/* Main Content */}
        <main className="flex-1 pb-20 md:pb-0 overflow-y-auto">
          {children}
        </main>
      </div>

      {/* Optimized Mobile Bottom Nav */}
      <nav className="md:hidden fixed bottom-0 left-0 right-0 border-t border-white/5 bg-zinc-950/80 backdrop-blur-lg px-8 py-3 flex justify-between items-center z-50">
        {navItems.map((item) => (
          <Link
            key={item.name}
            href={item.href}
            prefetch={true} // FOR SPEED
            className={cn(
              "flex flex-col items-center gap-1 transition-all",
              pathname === item.href ? "text-white scale-110" : "text-zinc-500 opacity-50"
            )}
          >
            <item.icon className="w-6 h-6" />
            <span className="text-[10px] font-bold uppercase tracking-tighter">{item.name}</span>
          </Link>
        ))}
        <button onClick={handleLogout} className="text-zinc-500 opacity-50">
          <LogOut className="w-6 h-6" />
        </button>
      </nav>
    </div>
  )
}