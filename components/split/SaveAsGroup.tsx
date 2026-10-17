"use client";
import { useState } from "react";
import { Check, Loader2, Users } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type { SplitSession } from "@/hooks/use-split-session";

// On the Summary: keep these people as a saved group for next time. It is offered after the split, never
// before it, so starting a split is never blocked by naming a group.
export function SaveAsGroup({ s }: { s: SplitSession }) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState(s.sessionName.trim());
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (s.activeGroupId && s.groupName) {
    return (
      <p data-testid="group-saved" className="flex min-h-11 items-center justify-center gap-2 text-sm text-muted-foreground">
        <Check size={14} className="text-success" /> Saved as group: {s.groupName}
      </p>
    );
  }
  if (!s.canSaveAsGroup) return null;

  if (!open) {
    return (
      <Button type="button" variant="glass" data-testid="save-group-open" className="h-12 w-full rounded-xl text-sm font-semibold" onClick={() => setOpen(true)}>
        <Users className="mr-2 size-4" /> Save these people as a group
      </Button>
    );
  }

  const save = async () => {
    setError(null);
    setSaving(true);
    const message = await s.saveAsGroup(name);
    setSaving(false);
    if (message) setError(message);
  };

  return (
    <div className="space-y-2 rounded-2xl glass-inset p-3">
      <div className="flex gap-2">
        <Input
          aria-label="Group name"
          data-testid="save-group-name"
          placeholder="Group name (e.g. Weekend Crew)"
          maxLength={60}
          value={name}
          autoFocus
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && name.trim() && (e.preventDefault(), void save())}
        />
        <Button type="button" data-testid="save-group-confirm" onClick={save} disabled={!name.trim() || saving} className="h-12 shrink-0 px-5">
          {saving ? <Loader2 className="size-4 animate-spin" /> : "Save"}
        </Button>
      </div>
      {error && (
        <p role="alert" className="text-sm text-coral">
          {error}
        </p>
      )}
    </div>
  );
}
