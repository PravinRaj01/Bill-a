import { AppShell } from "@/components/app-shell"
import { auth } from "@/lib/auth"

export default async function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  // Resolved on the server from the signed session cookie (no DB query — JWT
  // sessions), so guests never see History / Account / Logout links.
  const signedIn = !!(await auth())?.user?.id
  return <AppShell signedIn={signedIn}>{children}</AppShell>
}
