"use client";
import { useState } from "react";
import { Loader2, Plus, Trash2, X } from "lucide-react";

import { PersonAvatar } from "@/components/person-avatar";
import { Button } from "@/components/ui/button";
import { Drawer, DrawerContent, DrawerDescription, DrawerHeader, DrawerTitle } from "@/components/ui/drawer";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import type { SavedGroup } from "@/lib/db/schema";

const MAX_PEOPLE = 50;
const same = (a: string, b: string) => a.trim().toLowerCase() === b.trim().toLowerCase();

// Rename a saved group and change who is in it. Nothing is saved until Save; the sheet is remounted per
// group (see the key at the call site), so its fields always start from that group.
export function GroupEditSheet({
  group,
  open,
  onOpenChange,
  onSave,
  onDelete,
}: {
  group: SavedGroup;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Resolves to an error message, or null when saved. */
  onSave: (input: { groupName: string; names: string[] }) => Promise<string | null>;
  onDelete: () => void;
}) {
  const [name, setName] = useState(group.groupName);
  const [people, setPeople] = useState<string[]>(group.names);
  const [newPerson, setNewPerson] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const add = () => {
    const n = newPerson.trim();
    if (!n) return;
    if (people.some((p) => same(p, n))) {
      setError(`${n} is already in this group.`);
      return;
    }
    if (people.length >= MAX_PEOPLE) {
      setError(`A group can have up to ${MAX_PEOPLE} people.`);
      return;
    }
    setError(null);
    setPeople([...people, n]);
    setNewPerson("");
  };

  const save = async () => {
    setError(null);
    setSaving(true);
    const message = await onSave({ groupName: name.trim(), names: people });
    setSaving(false);
    if (message) setError(message);
    else onOpenChange(false);
  };

  const canSave = name.trim().length > 0 && people.length > 0 && !saving;

  return (
    <Drawer open={open} onOpenChange={onOpenChange}>
      <DrawerContent className="border-white/10 bg-surface text-white" data-testid="group-edit">
        <DrawerHeader>
          <DrawerTitle className="text-white">Edit group</DrawerTitle>
          <DrawerDescription className="text-muted-foreground">Rename it, or change who is in it.</DrawerDescription>
        </DrawerHeader>

        <div className="mx-auto max-h-[70vh] w-full max-w-md space-y-5 overflow-y-auto px-4 pb-6" data-vaul-no-drag>
          <div className="space-y-1.5">
            <Label htmlFor="group-name" className="text-xs font-semibold text-muted-foreground">
              Group name
            </Label>
            <Input id="group-name" data-testid="group-name" maxLength={60} value={name} onChange={(e) => setName(e.target.value)} className="h-12 rounded-xl border-white/10 bg-surface-2 text-base text-white" />
          </div>

          <div className="space-y-2">
            <p className="text-xs font-semibold text-muted-foreground">
              People <span className="tabular-nums">({people.length})</span>
            </p>
            <ul className="flex flex-wrap gap-2" data-testid="group-members">
              {people.map((p) => (
                <li key={p} className="flex items-center gap-2 rounded-full bg-surface-2 py-1 pr-1 pl-1">
                  <PersonAvatar name={p} compact className="size-7" />
                  <span className="text-sm font-semibold">{p}</span>
                  <button
                    type="button"
                    aria-label={`Remove ${p}`}
                    onClick={() => setPeople(people.filter((x) => x !== p))}
                    className="flex size-8 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-white/10 hover:text-white"
                  >
                    <X size={14} />
                  </button>
                </li>
              ))}
              {people.length === 0 && <li className="text-sm text-muted-foreground">Add at least one person.</li>}
            </ul>
            <div className="flex gap-2">
              <Input
                data-testid="group-add-input"
                aria-label="Add a person"
                placeholder="Add a person"
                maxLength={60}
                value={newPerson}
                onChange={(e) => setNewPerson(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), add())}
                enterKeyHint="done"
                className="h-12 rounded-xl border-white/10 bg-surface-2 text-base text-white"
              />
              <Button type="button" data-testid="group-add" variant="outline" onClick={add} disabled={!newPerson.trim()} className="h-12 rounded-xl border-white/10 px-4" aria-label="Add person">
                <Plus size={18} />
              </Button>
            </div>
          </div>

          {error && (
            <p role="alert" data-testid="group-error" className="text-sm text-coral">
              {error}
            </p>
          )}

          <div className="flex gap-3">
            <Button type="button" data-testid="group-save" onClick={save} disabled={!canSave} className="h-12 flex-1 rounded-xl bg-coral text-base font-bold text-coral-foreground hover:bg-coral/90">
              {saving && <Loader2 className="mr-2 size-4 animate-spin" />} Save
            </Button>
            <Button type="button" data-testid="group-delete" variant="outline" onClick={onDelete} aria-label="Delete group" className="h-12 rounded-xl border-white/10 px-4 text-red-400 hover:bg-red-500/10 hover:text-red-300">
              <Trash2 size={18} />
            </Button>
          </div>
        </div>
      </DrawerContent>
    </Drawer>
  );
}
