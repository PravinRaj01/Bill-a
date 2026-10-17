'use client'
import { useState, useEffect } from "react"
import Link from "next/link"
import { ChevronLeft, Pencil, Users } from "lucide-react"
import { toast } from "sonner"

import { GroupEditSheet } from "@/components/groups/GroupEditSheet"
import { PageTransition } from "@/components/page-transition"
import { AvatarStack } from "@/components/person-avatar"
import { ConfirmSheet } from "@/components/ui/confirm-sheet"
import { deleteGroup as deleteGroupAction, listGroups, updateGroup } from "@/lib/actions/groups"
import { getCurrentUser } from "@/lib/actions/user"
import type { SavedGroup } from "@/lib/db/schema"

export default function ManageGroups() {
  const [groups, setGroups] = useState<SavedGroup[]>([])
  // null = still loading, false = guest, true = signed in
  const [signedIn, setSignedIn] = useState<boolean | null>(null)
  const [editing, setEditing] = useState<SavedGroup | null>(null)
  const [deleting, setDeleting] = useState<SavedGroup | null>(null)

  useEffect(() => {
    getCurrentUser().then(async (user) => {
      setSignedIn(!!user)
      if (user) setGroups(await listGroups())
    })
  }, [])

  const save = async (g: SavedGroup, input: { groupName: string; names: string[] }): Promise<string | null> => {
    try {
      const r = await updateGroup(g.id, input)
      if (!r.ok) return r.error
      setGroups((all) => all.map((x) => (x.id === g.id ? r.group : x)))
      toast.success("Group saved")
      return null
    } catch {
      return "Couldn't save. Check your connection and try again."
    }
  }

  const remove = async (g: SavedGroup) => {
    try {
      // Server-side ownership check: only the signed-in user's own group can be deleted.
      await deleteGroupAction(g.id)
      setGroups((all) => all.filter((x) => x.id !== g.id))
      setEditing(null)
      toast("Group deleted")
    } catch {
      toast.error("Couldn't delete that group. Please try again.")
    }
  }

  return (
    <PageTransition>
      <div className="mx-auto mb-20 min-h-screen max-w-xl space-y-6 p-6 text-foreground md:p-10">
        <Link href="/dashboard" transitionTypes={["nav-back"]} className="flex min-h-11 items-center text-sm font-semibold text-muted-foreground transition-colors hover:text-white">
          <ChevronLeft size={16} className="mr-1" /> Home
        </Link>

        <header className="space-y-1">
          <h1 className="text-3xl font-black tracking-tighter text-white uppercase italic">Groups</h1>
          <p className="font-mono text-xs tracking-widest text-muted-foreground uppercase">Your saved crews</p>
        </header>

        <ul className="space-y-3">
          {groups.map((group) => (
            <li key={group.id} data-testid="group-card" className="space-y-4 rounded-3xl border border-white/5 bg-surface p-5">
              <div className="flex items-center gap-3">
                <AvatarStack names={group.names} max={4} />
                <div className="min-w-0 flex-1">
                  <h3 className="truncate text-base font-bold text-white">{group.groupName}</h3>
                  <p className="truncate text-sm text-muted-foreground">
                    {group.names.length} {group.names.length === 1 ? "person" : "people"} · {group.names.join(", ")}
                  </p>
                </div>
                <button
                  type="button"
                  data-testid="group-edit-open"
                  aria-label={`Edit ${group.groupName}`}
                  onClick={() => setEditing(group)}
                  className="flex size-11 shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-white/5 hover:text-white"
                >
                  <Pencil size={18} />
                </button>
              </div>
              <Link
                href={`/dashboard/new?group_id=${group.id}`}
                transitionTypes={["nav-forward"]}
                data-testid="group-split"
                className="flex min-h-11 w-full items-center justify-center rounded-xl border border-white/10 text-sm font-semibold text-zinc-200 transition-colors hover:bg-white/5"
              >
                Split with this group
              </Link>
            </li>
          ))}
        </ul>

        {groups.length === 0 && (
          <div className="space-y-3 rounded-3xl border-2 border-dashed border-white/5 px-6 py-14 text-center">
            <Users className="mx-auto size-9 text-muted-foreground/50" />
            {signedIn === false ? (
              <p className="text-sm text-muted-foreground">
                <Link href="/" className="font-semibold text-white underline underline-offset-4">Log in</Link> to save and reuse groups.
              </p>
            ) : signedIn === null ? (
              <p className="text-sm text-muted-foreground">Loading…</p>
            ) : (
              <>
                <p className="text-base font-semibold text-white">No saved groups yet</p>
                <p className="mx-auto max-w-xs text-sm text-muted-foreground">Add your people on the next split and choose &ldquo;Save this group&rdquo; to find them here.</p>
              </>
            )}
          </div>
        )}

        {editing && (
          <GroupEditSheet
            key={editing.id}
            group={editing}
            open
            onOpenChange={(o) => !o && setEditing(null)}
            onSave={(input) => save(editing, input)}
            onDelete={() => {
              toast.dismiss() // a leftover toast would sit on top of the sheet's Delete button
              setDeleting(editing)
            }}
          />
        )}

        <ConfirmSheet
          open={!!deleting}
          onOpenChange={(o) => !o && setDeleting(null)}
          title={`Delete ${deleting?.groupName ?? "this group"}?`}
          description="The saved group is removed. Bills you already split with it stay in History."
          confirmLabel="Delete"
          onConfirm={async () => {
            if (deleting) await remove(deleting)
          }}
        />
      </div>
    </PageTransition>
  )
}
