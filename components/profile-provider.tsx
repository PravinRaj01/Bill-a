"use client";
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

import { setTimezone } from "@/lib/actions/user";
import { meNameFor, samePerson } from "@/lib/split/participants";

// The signed-in user's own profile, available to every dashboard screen: the name they go by in splits, their
// @username and the portrait they picked. It starts from the session (the layout reads it without a database
// call) and is updated in place when they edit it, so every avatar changes at once.

export interface Profile {
  name: string | null;
  username: string | null;
  avatarSeed: string | null;
}

interface ProfileContext extends Profile {
  /** What the user is called inside a split: their first name, or "Me". */
  meName: string;
  signedIn: boolean;
  update: (patch: Partial<Profile>) => void;
}

const Ctx = createContext<ProfileContext>({ name: null, username: null, avatarSeed: null, meName: "Me", signedIn: false, update: () => {} });

const TZ_KEY = "billa.tz.v1";

export function ProfileProvider({ initial, signedIn, children }: { initial: Profile; signedIn: boolean; children: ReactNode }) {
  const [profile, setProfile] = useState<Profile>(initial);
  const update = useCallback((patch: Partial<Profile>) => setProfile((p) => ({ ...p, ...patch })), []);

  // Remember the browser's time zone, quietly and only when it changes: the daily digest needs the user's own day.
  useEffect(() => {
    if (!signedIn) return;
    try {
      const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;
      if (tz && localStorage.getItem(TZ_KEY) !== tz) void setTimezone(tz).then(() => localStorage.setItem(TZ_KEY, tz)).catch(() => {});
    } catch {
      /* storage or Intl unavailable: the default zone stays */
    }
  }, [signedIn]);

  const value = useMemo<ProfileContext>(() => ({ ...profile, meName: meNameFor(profile.name), signedIn, update }), [profile, signedIn, update]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export const useProfile = () => useContext(Ctx);

/** The portrait seed for this name when it is the user (their chosen one), else undefined (generated from the name). */
export function useSeedFor(name: string): string | undefined {
  const p = useProfile();
  return p.signedIn && p.avatarSeed && samePerson(name, p.meName) ? p.avatarSeed : undefined;
}
