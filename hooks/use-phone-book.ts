"use client";
import { useCallback, useEffect, useMemo, useState } from "react";

import { listPeople, type PersonView } from "@/lib/actions/people";
import { readCachedPeople, writeCachedPeople } from "@/lib/people/cache";

// The signed-in user's contacts, loaded once per page. Guests get an empty list. `phoneFor` finds a number by
// name (ignoring case and spacing), which is how a nudge knows to open that person's WhatsApp chat directly.
const key = (name: string) => name.trim().toLowerCase();

export function usePhoneBook() {
  const [people, setPeople] = useState<PersonView[] | null>(null);

  const reload = useCallback(async () => {
    try {
      const fresh = await listPeople();
      setPeople(fresh);
      void writeCachedPeople(fresh); // the copy the picker uses offline
    } catch {
      // Offline: the last saved list, so the picker still has your pals. Without one, nudges fall back to the share sheet.
      const cached = await readCachedPeople<PersonView>();
      setPeople((p) => p ?? cached ?? []);
    }
  }, []);

  useEffect(() => {
    // Show the saved copy straight away; the server answer replaces it as soon as it arrives.
    void readCachedPeople<PersonView>().then((cached) => cached && setPeople((p) => p ?? cached));
    // eslint-disable-next-line react-hooks/set-state-in-effect -- loading from the server on mount
    void reload();
  }, [reload]);

  const phones = useMemo(() => new Map((people ?? []).filter((p) => p.phone).map((p) => [key(p.name), p.phone as string])), [people]);
  const phoneFor = useCallback((name: string) => phones.get(key(name)) ?? null, [phones]);

  return { people, setPeople, reload, phoneFor };
}
