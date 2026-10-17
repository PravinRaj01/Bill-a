"use client";
import { useEffect, useState } from "react";
import { Check, Loader2, Shuffle } from "lucide-react";
import { toast } from "sonner";

import { useProfile } from "@/components/profile-provider";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { updateProfile, updateProfileName } from "@/lib/actions/user";
import { meNameFor } from "@/lib/split/participants";
import { avatarDataUri } from "@/lib/ui/avatar";
import { personColor } from "@/lib/ui/person-color";
import { parseProfileInput } from "@/lib/validation/profile";
import { cn } from "@/lib/utils";

// A starter set (so the grid is the same until you shuffle), then random ones. "" means the portrait made from
// your name, which is what everyone gets until they choose.
const STARTERS = ["sun", "moon", "star", "wave", "leaf", "fox", "owl", "pine", "dune", "mint", "ember"];
const randomSeeds = () => Array.from({ length: STARTERS.length }, () => Math.random().toString(36).slice(2, 8));

/** One portrait, drawn for `colorName` (the background follows the person's colour) from `seed`. */
function Portrait({ colorName, seed, className }: { colorName: string; seed?: string; className?: string }) {
  const [made, setMade] = useState<{ key: string; uri: string } | null>(null);
  const key = `${colorName}|${seed ?? ""}`;
  useEffect(() => {
    let cancelled = false;
    avatarDataUri(colorName, seed)
      .then((uri) => !cancelled && setMade({ key, uri }))
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [colorName, seed, key]);
  return made?.key === key ? (
    // eslint-disable-next-line @next/next/no-img-element
    <img src={made.uri} alt="" className={cn("size-full rounded-full object-cover", className)} />
  ) : (
    <span className="block size-full animate-pulse rounded-full bg-white/10" />
  );
}

// Account > Profile: how you appear to others. Your name (used in splits), the portrait you pick, and the
// @username friends will find you by.
export function ProfileEditor() {
  const profile = useProfile();
  const [name, setName] = useState(profile.name ?? "");
  const [username, setUsername] = useState(profile.username ?? "");
  const [seed, setSeed] = useState<string>(profile.avatarSeed ?? "");
  const [options, setOptions] = useState<string[]>(() => {
    // Keep the saved portrait in the grid even if it came from an earlier shuffle.
    const rest = STARTERS.filter((s) => s !== profile.avatarSeed);
    return profile.avatarSeed && !STARTERS.includes(profile.avatarSeed) ? [profile.avatarSeed, ...rest.slice(0, STARTERS.length - 1)] : STARTERS;
  });
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const meName = meNameFor(name);
  const dirty = name.trim() !== (profile.name ?? "") || username.trim().toLowerCase() !== (profile.username ?? "") || seed !== (profile.avatarSeed ?? "");

  const shuffle = () => {
    const next = randomSeeds();
    setOptions(seed && !next.includes(seed) ? [seed, ...next.slice(0, next.length - 1)] : next);
  };

  const save = async () => {
    setError(null);
    const wantUsername = username.trim().toLowerCase();
    // Check the shape here so the message appears at once; the server checks again (and the uniqueness).
    const parsed = parseProfileInput({ username: wantUsername || null, avatarSeed: seed || null });
    if (!parsed.ok) return setError(parsed.error);
    setSaving(true);
    try {
      if (name.trim() !== (profile.name ?? "")) {
        const r = await updateProfileName(name);
        if (!r.ok) return setError(r.error);
        profile.update({ name: r.name });
      }
      const r = await updateProfile({ username: wantUsername || null, avatarSeed: seed || null });
      if (!r.ok) return setError(r.error);
      profile.update({ username: r.username, avatarSeed: r.avatarSeed });
      setUsername(r.username ?? "");
      toast.success("Profile saved");
    } catch {
      setError("Couldn't save. Check your connection and try again.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div data-testid="profile-editor" className="glass space-y-6 rounded-3xl p-5">
      <section className="flex flex-col items-center gap-3 text-center">
        <div data-testid="profile-preview" className="size-28 rounded-full p-1 ring-4" style={{ ["--tw-ring-color" as string]: personColor(meName) }}>
          <Portrait colorName={meName} seed={seed || undefined} />
        </div>
        <p className="text-sm text-muted-foreground">
          In splits you appear as <strong className="text-foreground">{meName}</strong>
          {username.trim() && <> · @{username.trim().toLowerCase()}</>}
        </p>
      </section>

      <section className="space-y-3" aria-labelledby="pick-portrait">
        <div className="flex items-center justify-between">
          <h3 id="pick-portrait" className="text-sm font-bold">
            Your portrait
          </h3>
          <Button type="button" variant="ghost" size="sm" data-testid="portrait-shuffle" onClick={shuffle}>
            <Shuffle /> Shuffle
          </Button>
        </div>
        <ul role="radiogroup" aria-label="Portraits" className="grid grid-cols-4 gap-3">
          {["", ...options].map((s) => {
            const active = s === seed;
            return (
              <li key={s || "from-name"}>
                <button
                  type="button"
                  role="radio"
                  aria-checked={active}
                  aria-label={s ? `Portrait ${s}` : "Made from my name"}
                  data-testid={`portrait-${s || "default"}`}
                  onClick={() => setSeed(s)}
                  className={cn(
                    "relative aspect-square w-full rounded-full p-0.5 outline-none transition-transform focus-visible:ring-2 focus-visible:ring-white/40 active:scale-95",
                    active ? "bg-primary shadow-[var(--coral-glow)]" : "bg-white/10 hover:bg-white/20",
                  )}
                >
                  <Portrait colorName={meName} seed={s || undefined} />
                  {active && (
                    <span className="absolute -right-0.5 -bottom-0.5 flex size-5 items-center justify-center rounded-full bg-primary text-primary-foreground ring-2 ring-background">
                      <Check size={12} strokeWidth={3} />
                    </span>
                  )}
                </button>
              </li>
            );
          })}
        </ul>
        <p className="text-xs text-muted-foreground">The first one is made from your name. Friends see the one you pick.</p>
      </section>

      <section className="space-y-3">
        <div className="space-y-1.5">
          <Label htmlFor="profile-name" className="text-xs text-muted-foreground">
            Your name
          </Label>
          <Input id="profile-name" data-testid="profile-name" maxLength={80} autoComplete="name" value={name} onChange={(e) => setName(e.target.value)} />
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="profile-username" className="text-xs text-muted-foreground">
            Username
          </Label>
          <div className="relative">
            <span aria-hidden className="pointer-events-none absolute top-1/2 left-4 -translate-y-1/2 text-muted-foreground">
              @
            </span>
            <Input
              id="profile-username"
              data-testid="profile-username"
              className="pl-8"
              maxLength={20}
              autoCapitalize="none"
              autoCorrect="off"
              spellCheck={false}
              placeholder="alexis_r"
              value={username}
              onChange={(e) => setUsername(e.target.value.replace(/[^A-Za-z0-9_]/g, ""))}
            />
          </div>
          <p className="text-xs text-muted-foreground">3 to 20 letters, numbers or underscores. Friends will find you by this.</p>
        </div>
      </section>

      {error && (
        <p role="alert" data-testid="profile-error" className="text-sm text-coral">
          {error}
        </p>
      )}
      <Button type="button" size="lg" data-testid="profile-save" onClick={save} disabled={saving || !dirty} className="w-full">
        {saving && <Loader2 className="animate-spin" />} Save profile
      </Button>
    </div>
  );
}
