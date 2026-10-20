import { AppShell } from "@/components/app-shell"
import { SyncManager } from "@/components/sync-manager"
import { PushLive } from "@/components/notifications/PushLive"
import { PushResync } from "@/components/notifications/PushResync"
import { ProfileProvider } from "@/components/profile-provider"
import { auth } from "@/lib/auth"

export default async function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  // Resolved on the server from the signed session cookie (no DB query — JWT
  // sessions), so guests never see History / Account / Logout links.
  const user = (await auth())?.user
  const userId = user?.id ?? null
  return (
    <ProfileProvider signedIn={!!userId} initial={{ name: user?.name ?? null, username: user?.username ?? null, avatarSeed: user?.avatarSeed ?? null }}>
      <AppShell signedIn={!!userId}>
        <SyncManager userId={userId} />
        <PushResync signedIn={!!userId} />
        <PushLive signedIn={!!userId} />
        {children}
      </AppShell>
    </ProfileProvider>
  )
}
