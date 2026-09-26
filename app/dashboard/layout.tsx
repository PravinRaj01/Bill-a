import { AppShell } from "@/components/app-shell"
import { SyncManager } from "@/components/sync-manager"
import { auth } from "@/lib/auth"

export default async function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  // Resolved on the server from the signed session cookie (no DB query — JWT
  // sessions), so guests never see History / Account / Logout links.
  const userId = (await auth())?.user?.id ?? null
  return (
    <AppShell signedIn={!!userId}>
      <SyncManager userId={userId} />
      {children}
    </AppShell>
  )
}
