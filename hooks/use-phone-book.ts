"use client";
import { useCallback, useEffect, useMemo, useState } from "react";

import { listPeople, type PersonView } from "@/lib/actions/people";

// The signed-in user's contacts, loaded once per page. Guests get an empty list. `phoneFor` finds a number by
// name (ignoring case and spacing), which is how a nudge knows to open that person's WhatsApp chat directly.
const key = (name: string) => name.trim().toLowerCase();

export function usePhoneBook() {
  const [people, setPeople] = useState<PersonView[] | null>(null);

  const reload = useCallback(async () => {
    try {
      setPeople(await listPeople());
    } catch {
      setPeople((p) => p ?? []); // offline: nudges fall back to the share sheet
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- loading from the server on mount
    void reload();
  }, [reload]);

  const phones = useMemo(() => new Map((people ?? []).filter((p) => p.phone).map((p) => [key(p.name), p.phone as string])), [people]);
  const phoneFor = useCallback((name: string) => phones.get(key(name)) ?? null, [phones]);

  return { people, setPeople, reload, phoneFor };
}
