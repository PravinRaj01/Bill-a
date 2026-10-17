"use client";
import { useState } from "react";
import { ChevronDown, ChevronUp, Loader2, Plus, RefreshCw, Users, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { PersonAvatar } from "@/components/person-avatar";
import type { SplitSession } from "@/hooks/use-split-session";

export function NamesStep({ s }: { s: SplitSession }) {
  const [newName, setNewName] = useState("");
  const [showGroupList, setShowGroupList] = useState(false);

  const add = () => {
    if (s.addPerson(newName)) setNewName("");
  };
  const duplicateGroupName =
    s.saveThisGroup &&
    s.groupName.trim() !== "" &&
    s.savedGroups.some((g) => g.groupName.trim().toLowerCase() === s.groupName.trim().toLowerCase());

  return (
    <div className="space-y-6">
      <div className="space-y-1">
        <h2 className="type-title">Who&apos;s splitting?</h2>
        <p className="text-sm text-muted-foreground">Step {s.quick ? 2 : 1} of 3 · add everyone at the table</p>
      </div>

      <div className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="session-name" className="ml-1 text-xs font-semibold text-muted-foreground">
            Session name (optional)
          </Label>
          <Input
            id="session-name"
            placeholder="e.g. Friday Dinner"
            className="font-semibold"
            value={s.sessionName}
            onChange={(e) => s.setSessionName(e.target.value)}
          />
        </div>

        {!s.isGuest && s.savedGroups.length > 0 && (
          <div className="overflow-hidden rounded-2xl glass">
            <button
              type="button"
              onClick={() => setShowGroupList(!showGroupList)}
              aria-expanded={showGroupList}
              className="flex min-h-12 w-full items-center justify-between px-4 text-sm font-semibold text-muted-foreground transition-colors hover:bg-white/5 hover:text-foreground"
            >
              <span className="flex items-center gap-2"><Users size={16} /> Load a saved group</span>
              {showGroupList ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            </button>
            {showGroupList && (
              <div className="max-h-56 overflow-y-auto border-t border-white/8 glass-inset">
                {s.savedGroups.map((group) => (
                  <button
                    key={group.id}
                    type="button"
                    onClick={() => {
                      s.loadSavedGroup(group);
                      setShowGroupList(false);
                    }}
                    className="flex min-h-12 w-full items-center justify-between border-b border-white/8 px-4 text-left text-sm text-foreground last:border-0 hover:bg-white/10"
                  >
                    <span className="font-medium">{group.groupName}</span>
                    <span className="text-xs text-muted-foreground">{group.names.length} people</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      <div className="space-y-4 rounded-3xl glass p-5">
        <div className="flex gap-2">
          <Input
            aria-label="Person's name"
           
            placeholder="Enter name…"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && add()}
            enterKeyHint="done"
          />
          <Button
            type="button"
            aria-label="Add person"
            onClick={add}
            disabled={!newName.trim()}
            className="size-12 shrink-0"
          >
            <Plus size={20} />
          </Button>
        </div>

        <div className="flex min-h-14 flex-wrap gap-2 rounded-xl border border-white/8 glass-inset p-2">
          {s.people.length === 0 && <p className="self-center px-2 text-sm text-muted-foreground">Nobody yet</p>}
          {s.people.map((p) => (
            <span key={p} className="flex items-center gap-2 rounded-full bg-white/5 py-1 pl-1 pr-1 text-sm text-foreground">
              <PersonAvatar name={p} compact className="size-7" />
              {p}
              <button
                type="button"
                aria-label={`Remove ${p}`}
                onClick={() => s.removePerson(p)}
                className="flex size-8 items-center justify-center rounded-full transition-colors hover:bg-white/15"
              >
                <X size={14} />
              </button>
            </span>
          ))}
        </div>

        {!s.isGuest && (
          <div className="space-y-3 pt-1">
            {s.activeGroupId && s.hasGroupChanged() ? (
              <div className="rounded-xl border border-warning/20 bg-warning/10 p-3">
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2 text-sm font-semibold text-warning">
                    <RefreshCw size={14} /> Update saved group?
                  </div>
                  <Switch checked={s.updateGroup} onCheckedChange={s.setUpdateGroup} aria-label="Update saved group" className="data-[state=checked]:bg-warning" />
                </div>
                <p className="mt-1 pl-6 text-xs text-muted-foreground">
                  Update <strong>{s.groupName}</strong> with these changes?
                </p>
              </div>
            ) : !s.activeGroupId ? (
              <>
                <div className="flex min-h-11 items-center justify-between px-1">
                  <span className="text-sm font-medium text-foreground/80">Save this group?</span>
                  <Switch checked={s.saveThisGroup} onCheckedChange={s.setSaveThisGroup} aria-label="Save this group" />
                </div>
                {s.saveThisGroup && (
                  <Input
                    aria-label="Group name"
                    placeholder="Group name (e.g. Weekend Crew)"
                    value={s.groupName}
                    onChange={(e) => s.setGroupName(e.target.value)}
                   
                  />
                )}
                {duplicateGroupName && (
                  <p data-testid="group-exists" className="px-1 text-xs text-warning">
                    You already have a group called “{s.groupName.trim()}” — saving will update its members, not add another.
                  </p>
                )}
              </>
            ) : (
              <div className="flex items-center justify-center gap-2 p-2 text-sm text-muted-foreground">
                <Users size={14} /> Loaded: {s.groupName}
              </div>
            )}
          </div>
        )}

        <Button
          type="button"
          className="h-12 w-full"
          disabled={s.people.length < 1 || s.starting}
          onClick={s.startScanning}
        >
          {s.starting ? <Loader2 className="animate-spin" /> : s.quick ? "Continue" : "Start scanning"}
        </Button>
      </div>
    </div>
  );
}
