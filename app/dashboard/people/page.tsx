"use client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { ChevronLeft, Plus, UserRound } from "lucide-react";
import { toast } from "sonner";

import { AddPalSheet, type AddTab } from "@/components/people/AddPalSheet";
import { FriendRequests, friendLabel } from "@/components/people/FriendRequests";
import { LinkContactSheet } from "@/components/people/LinkContactSheet";
import { PastBillsSheet } from "@/components/people/PastBillsSheet";
import { PageTransition } from "@/components/page-transition";
import { PersonSheet } from "@/components/people/PersonSheet";
import { PersonAvatar } from "@/components/person-avatar";
import { Button } from "@/components/ui/button";
import { ConfirmSheet } from "@/components/ui/confirm-sheet";
import { usePaymentShare } from "@/hooks/use-payment-share";
import { usePhoneBook } from "@/hooks/use-phone-book";
import { getFriends, respondToRequest, unfriend, type FriendView, type FriendsView } from "@/lib/actions/friends";
import { addPerson, removePerson, suggestPeople, updatePerson, type PersonView } from "@/lib/actions/people";
import { getBalances, listBills, type BalancesResult, type BillListEntry } from "@/lib/actions/history";
import { getCurrentUser } from "@/lib/actions/user";
import { formatMoney } from "@/lib/money";
import { formatPhone } from "@/lib/people/phone";
import { balanceNudgeText, type PersonBalance } from "@/lib/split/settle";
import { shareMessage, shareToPerson } from "@/lib/ui/share";

const key = (name: string) => name.trim().toLowerCase();

type Flow = { id: string; name: string; step: "link" | "past" };

// Everyone you split with, in one list: a name and a WhatsApp number, and for pals on Bill.a their @username and
// portrait. Each row shows what they owe you (or you them), and the sheet has their latest bills and a one-tap nudge.
// Requests sit above the list. Becoming pals walks through two optional steps: link a contact you already had, then
// share the bills you have already split with them.
export default function PalsPage() {
  const { people, setPeople, reload } = usePhoneBook();
  const [signedIn, setSignedIn] = useState<boolean | null>(null);
  const [balances, setBalances] = useState<BalancesResult | null>(null);
  const [bills, setBills] = useState<BillListEntry[]>([]);
  const [suggestions, setSuggestions] = useState<{ name: string; count: number }[]>([]);
  const [friends, setFriends] = useState<FriendsView | null>(null);
  const [editing, setEditing] = useState<PersonView | null>(null);
  const [adding, setAdding] = useState<AddTab | null>(null);
  const [deleting, setDeleting] = useState<PersonView | null>(null);
  const [unfriending, setUnfriending] = useState<PersonView | null>(null);
  const [flow, setFlow] = useState<Flow | null>(null);
  const payment = usePaymentShare();

  const loadFriends = useCallback(async (): Promise<FriendsView> => {
    try {
      const f = await getFriends();
      setFriends(f);
      return f;
    } catch {
      const empty: FriendsView = { friends: [], incoming: [], outgoing: [] };
      setFriends((s) => s ?? empty);
      return empty;
    }
  }, []);

  useEffect(() => {
    getCurrentUser().then((u) => setSignedIn(!!u)).catch(() => setSignedIn(false));
    getBalances().then(setBalances).catch(() => setBalances({ people: [], tracked: 0, withoutMe: 0 }));
    listBills().then(setBills).catch(() => setBills([]));
    suggestPeople().then(setSuggestions).catch(() => setSuggestions([]));
    // eslint-disable-next-line react-hooks/set-state-in-effect -- load from the server on mount
    void loadFriends();
  }, [loadFriends]);

  // The + sheet opens this page ready to add (?add=person or ?add=friend), and a notification opens the share offer
  // for a new pal (?share=<their id>).
  const paramsRead = useRef(false);
  useEffect(() => {
    if (paramsRead.current || signedIn !== true || !friends) return;
    paramsRead.current = true;
    const q = new URLSearchParams(window.location.search);
    const add = q.get("add");
    const share = q.get("share");
    const pal = share ? friends.friends.find((f) => f.id === share) : undefined;
    queueMicrotask(() => {
      if (add === "person") setAdding("name");
      if (add === "friend") setAdding("friend");
      if (pal) setFlow({ id: pal.id, name: friendLabel(pal), step: "link" });
    });
    if (add || share) window.history.replaceState(null, "", window.location.pathname);
  }, [signedIn, friends]);

  const balanceFor = useCallback(
    (name: string, linkedUserId?: string | null) => (balances?.people ?? []).filter((b) => (linkedUserId && b.userId === linkedUserId) || key(b.name) === key(name)),
    [balances],
  );
  const recentFor = useCallback(
    (name: string, linkedUserId?: string | null) =>
      bills
        .filter((b) => b.people.some((p) => key(p) === key(name)) || (!!linkedUserId && b.linkedIds.includes(linkedUserId)))
        .slice(0, 5)
        .map((b) => ({ id: b.id, title: b.billTitle, createdAt: b.createdAt })),
    [bills],
  );

  const sorted = useMemo(() => [...(people ?? [])].sort((a, b) => a.name.localeCompare(b.name)), [people]);

  // Keep what the list knows and an edit does not return (the friend behind a linked contact).
  const upsert = (p: PersonView) => setPeople((all) => [...(all ?? []).filter((x) => x.id !== p.id), { ...(all ?? []).find((x) => x.id === p.id), ...p }]);

  const add = async (name: string, phone: string): Promise<string | null> => {
    try {
      const r = await addPerson({ name, phone });
      if (!r.ok) return r.error;
      upsert(r.person);
      setSuggestions((s) => s.filter((x) => key(x.name) !== key(name)));
      toast.success(`${r.person.name} added`);
      return null;
    } catch {
      return "Couldn't save. Check your connection and try again.";
    }
  };

  const save = async (p: PersonView, input: { name: string; phone: string }): Promise<string | null> => {
    try {
      const r = await updatePerson(p.id, input);
      if (!r.ok) return r.error;
      upsert(r.person);
      toast.success("Saved");
      return null;
    } catch {
      return "Couldn't save. Check your connection and try again.";
    }
  };

  const remove = async (p: PersonView) => {
    try {
      await removePerson(p.id);
      setPeople((all) => (all ?? []).filter((x) => x.id !== p.id));
      setEditing(null);
      toast("Pal removed");
    } catch {
      toast.error("Couldn't remove that pal. Please try again.");
    }
  };

  const nudge = async (p: PersonView, b: PersonBalance) => {
    const text = balanceNudgeText({
      name: b.name,
      payTo: "me",
      currency: b.currency,
      lines: b.bills.filter((x) => x.amount > 0).map((x) => ({ title: x.title, date: x.createdAt, amount: x.amount })),
      payment: payment.block || undefined,
    });
    const msg = shareMessage(await shareToPerson(text, p.phone));
    if (msg) toast(msg);
  };

  // --- becoming pals ------------------------------------------------------------------------------------
  const startFlow = (pal: FriendView) => setFlow({ id: pal.id, name: friendLabel(pal), step: "link" });
  const toPast = useCallback(() => {
    void reload();
    setFlow((f) => (f ? { ...f, step: "past" } : f));
  }, [reload]);
  const endFlow = useCallback(() => setFlow(null), []);

  const answer = async (u: FriendView, accept: boolean) => {
    try {
      await respondToRequest(u.id, accept);
      await loadFriends();
      await reload();
      if (accept) startFlow(u);
    } catch {
      toast.error("Couldn't answer that. Please try again.");
    }
  };

  const cancelRequest = async (u: FriendView) => {
    try {
      await unfriend(u.id);
      await loadFriends();
      toast("Request cancelled");
    } catch {
      toast.error("Couldn't cancel that. Please try again.");
    }
  };

  const sent = async (status: "requested" | "accepted", username: string) => {
    const f = await loadFriends();
    if (status !== "accepted") return;
    await reload();
    const pal = f.friends.find((x) => x.username === username);
    if (pal) startFlow(pal);
  };

  const endFriendship = async (p: PersonView) => {
    if (!p.linkedUserId) return;
    try {
      await unfriend(p.linkedUserId);
      await loadFriends();
      await reload();
      setEditing(null);
      toast("Unfriended. Bills you shared are no longer visible to each other.");
    } catch {
      toast.error("Couldn't do that. Please try again.");
    }
  };

  return (
    <PageTransition>
      <div className="mx-auto mb-20 min-h-screen max-w-xl space-y-6 p-6 text-foreground md:p-10">
        <Link href="/dashboard" transitionTypes={["nav-back"]} className="flex min-h-11 items-center text-sm font-semibold text-muted-foreground transition-colors hover:text-foreground">
          <ChevronLeft size={16} className="mr-1" /> Home
        </Link>

        <header className="flex items-end justify-between gap-3">
          <div className="space-y-1">
            <h1 className="type-title">Pals</h1>
            <p className="type-eyebrow">Who you split with</p>
          </div>
          {signedIn && (
            <Button type="button" data-testid="person-add-open" onClick={() => setAdding("friend")} className="h-11 rounded-full px-4">
              <Plus className="mr-1 size-4" /> Add
            </Button>
          )}
        </header>

        {signedIn && <FriendRequests state={friends} onAnswer={answer} onCancel={cancelRequest} />}

        {signedIn && suggestions.length > 0 && (
          <section aria-label="Pals from your bills" className="space-y-2">
            <p className="type-eyebrow">From your bills</p>
            <ul className="flex flex-wrap gap-2">
              {suggestions.map((s) => (
                <li key={s.name}>
                  <button
                    type="button"
                    data-testid="person-suggestion"
                    onClick={() => void add(s.name, "")}
                    className="flex min-h-11 items-center gap-2 rounded-full glass py-1 pr-4 pl-1 text-sm font-semibold transition-colors hover:bg-white/10"
                  >
                    <PersonAvatar name={s.name} compact className="size-8" />
                    {s.name}
                    <Plus size={14} className="text-muted-foreground" />
                  </button>
                </li>
              ))}
            </ul>
          </section>
        )}

        <ul className="space-y-3" data-testid="pal-list">
          {sorted.map((p) => {
            const owed = balanceFor(p.name, p.linkedUserId);
            return (
              <li key={p.id}>
                <button
                  type="button"
                  data-testid="person-row"
                  data-friend={p.linkedUserId ? "true" : "false"}
                  onClick={() => setEditing(p)}
                  className="flex min-h-16 w-full items-center gap-3 rounded-3xl glass p-4 text-left transition-colors hover:bg-white/10"
                >
                  <PersonAvatar name={p.name} seed={p.friend?.avatarSeed} className="size-11" />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-base font-bold">{p.name}</span>
                    <span className="block truncate text-sm text-muted-foreground">
                      {p.friend ? `${p.friend.username ? `@${p.friend.username}` : "On Bill.a"}${p.phone ? ` · ${formatPhone(p.phone)}` : ""}` : p.phone ? formatPhone(p.phone) : "No number yet"}
                    </span>
                  </span>
                  <span className="shrink-0 text-right text-xs">
                    {owed.map((b) => (
                      <span key={b.key} data-testid="person-chip" className={`block font-bold tabular-nums ${b.net > 0 ? "text-success" : "text-coral"}`}>
                        {b.net > 0 ? "owes " : "you owe "}
                        {formatMoney(Math.abs(b.net), b.currency)}
                      </span>
                    ))}
                    {p.linkedUserId && <span data-testid="friend-badge" className="mt-0.5 block text-[10px] font-semibold tracking-wide text-muted-foreground uppercase">Pal</span>}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>

        {people !== null && sorted.length === 0 && (
          <div className="space-y-3 rounded-3xl border-2 border-dashed border-white/8 px-6 py-14 text-center">
            <UserRound className="mx-auto size-9 text-muted-foreground/50" />
            {signedIn === false ? (
              <p className="text-sm text-muted-foreground">
                <Link href="/" className="font-semibold text-foreground underline underline-offset-4">Log in</Link> to keep your pals and nudge them on WhatsApp.
              </p>
            ) : (
              <>
                <p className="text-base font-semibold text-foreground">No pals yet</p>
                <p className="mx-auto max-w-xs text-sm text-muted-foreground">Add the people you split with, with their WhatsApp number, and nudges go straight to their chat. Find friends on Bill.a by @username.</p>
              </>
            )}
          </div>
        )}

        {adding && (
          <AddPalSheet
            key={adding}
            open
            initialTab={adding}
            onOpenChange={(o) => !o && setAdding(null)}
            onSent={(status, username) => void sent(status, username)}
            onAddName={(i) => add(i.name, i.phone)}
          />
        )}

        {editing && (
          <PersonSheet
            key={editing.id}
            person={editing}
            balances={balanceFor(editing.name, editing.linkedUserId)}
            recent={recentFor(editing.name, editing.linkedUserId)}
            open
            onOpenChange={(o) => !o && setEditing(null)}
            onSave={(input) => save(editing, input)}
            onDelete={() => {
              toast.dismiss(); // a leftover toast would sit on top of the sheet's Delete button
              setDeleting(editing);
            }}
            onNudge={(b) => void nudge(editing, b)}
            onSharePast={() => {
              const id = editing.linkedUserId;
              if (!id) return;
              setEditing(null);
              setFlow({ id, name: editing.name, step: "past" });
            }}
            onUnfriend={() => setUnfriending(editing)}
          />
        )}

        <LinkContactSheet friend={flow?.step === "link" ? flow : null} onDone={toPast} />
        <PastBillsSheet friend={flow?.step === "past" ? flow : null} onClose={endFlow} />

        <ConfirmSheet
          open={!!deleting}
          onOpenChange={(o) => !o && setDeleting(null)}
          title={`Remove ${deleting?.name ?? "this pal"}?`}
          description="They're removed from your pals. Bills you've already split with them stay in History."
          confirmLabel="Remove"
          onConfirm={async () => {
            if (deleting) await remove(deleting);
          }}
        />
        <ConfirmSheet
          open={!!unfriending}
          onOpenChange={(o) => !o && setUnfriending(null)}
          title={`Unfriend ${unfriending?.name ?? "this pal"}?`}
          description="You stop seeing each other's shared bills right away. They stay in your pals as a plain name."
          confirmLabel="Unfriend"
          onConfirm={async () => {
            if (unfriending) await endFriendship(unfriending);
          }}
        />
      </div>
    </PageTransition>
  );
}
