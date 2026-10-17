"use client";
import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { ChevronLeft, Plus, UserRound } from "lucide-react";
import { toast } from "sonner";

import { PageTransition } from "@/components/page-transition";
import { PersonSheet } from "@/components/people/PersonSheet";
import { PersonAvatar } from "@/components/person-avatar";
import { Button } from "@/components/ui/button";
import { ConfirmSheet } from "@/components/ui/confirm-sheet";
import { usePaymentShare } from "@/hooks/use-payment-share";
import { usePhoneBook } from "@/hooks/use-phone-book";
import { addPerson, removePerson, suggestPeople, updatePerson, type PersonView } from "@/lib/actions/people";
import { getBalances, listBills, type BalancesResult, type BillListEntry } from "@/lib/actions/history";
import { getCurrentUser } from "@/lib/actions/user";
import { formatMoney } from "@/lib/money";
import { formatPhone } from "@/lib/people/phone";
import { balanceNudgeText, type PersonBalance } from "@/lib/split/settle";
import { shareMessage, shareToPerson } from "@/lib/ui/share";

const key = (name: string) => name.trim().toLowerCase();

// Everyone you split with: a name, and a WhatsApp number if you have it. Each row shows what they owe you
// (or you them), and the sheet has their latest bills and a one-tap nudge.
export default function PeoplePage() {
  const { people, setPeople } = usePhoneBook();
  const [signedIn, setSignedIn] = useState<boolean | null>(null);
  const [balances, setBalances] = useState<BalancesResult | null>(null);
  const [bills, setBills] = useState<BillListEntry[]>([]);
  const [suggestions, setSuggestions] = useState<{ name: string; count: number }[]>([]);
  const [editing, setEditing] = useState<PersonView | "new" | null>(null);
  const [deleting, setDeleting] = useState<PersonView | null>(null);
  const payment = usePaymentShare();

  useEffect(() => {
    getCurrentUser().then((u) => setSignedIn(!!u)).catch(() => setSignedIn(false));
    getBalances().then(setBalances).catch(() => setBalances({ people: [], tracked: 0, withoutMe: 0 }));
    listBills().then(setBills).catch(() => setBills([]));
    suggestPeople().then(setSuggestions).catch(() => setSuggestions([]));
  }, []);

  const balanceFor = useCallback((name: string) => (balances?.people ?? []).filter((b) => key(b.name) === key(name)), [balances]);
  const recentFor = useCallback(
    (name: string) => bills.filter((b) => b.people.some((p) => key(p) === key(name))).slice(0, 5).map((b) => ({ id: b.id, title: b.billTitle, createdAt: b.createdAt })),
    [bills],
  );

  const sorted = useMemo(() => [...(people ?? [])].sort((a, b) => a.name.localeCompare(b.name)), [people]);

  const upsert = (p: PersonView) => setPeople((all) => [...(all ?? []).filter((x) => x.id !== p.id), p]);

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
      toast("Person removed");
    } catch {
      toast.error("Couldn't remove that person. Please try again.");
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

  return (
    <PageTransition>
      <div className="mx-auto mb-20 min-h-screen max-w-xl space-y-6 p-6 text-foreground md:p-10">
        <Link href="/dashboard" transitionTypes={["nav-back"]} className="flex min-h-11 items-center text-sm font-semibold text-muted-foreground transition-colors hover:text-foreground">
          <ChevronLeft size={16} className="mr-1" /> Home
        </Link>

        <header className="flex items-end justify-between gap-3">
          <div className="space-y-1">
            <h1 className="type-title">People</h1>
            <p className="type-eyebrow">Who you split with</p>
          </div>
          {signedIn && (
            <Button type="button" data-testid="person-add-open" onClick={() => setEditing("new")} className="h-11 rounded-full px-4">
              <Plus className="mr-1 size-4" /> Add
            </Button>
          )}
        </header>

        {signedIn && suggestions.length > 0 && (
          <section aria-label="People from your bills" className="space-y-2">
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

        <ul className="space-y-3">
          {sorted.map((p) => {
            const owed = balanceFor(p.name);
            return (
              <li key={p.id}>
                <button
                  type="button"
                  data-testid="person-row"
                  onClick={() => setEditing(p)}
                  className="flex min-h-16 w-full items-center gap-3 rounded-3xl glass p-4 text-left transition-colors hover:bg-white/10"
                >
                  <PersonAvatar name={p.name} className="size-11" />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-base font-bold">{p.name}</span>
                    <span className="block truncate text-sm text-muted-foreground">{p.phone ? formatPhone(p.phone) : "No number yet"}</span>
                  </span>
                  <span className="shrink-0 text-right text-xs">
                    {owed.map((b) => (
                      <span key={b.key} data-testid="person-chip" className={`block font-bold tabular-nums ${b.net > 0 ? "text-success" : "text-coral"}`}>
                        {b.net > 0 ? "owes " : "you owe "}
                        {formatMoney(Math.abs(b.net), b.currency)}
                      </span>
                    ))}
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
                <Link href="/" className="font-semibold text-foreground underline underline-offset-4">Log in</Link> to keep your people and nudge them on WhatsApp.
              </p>
            ) : (
              <>
                <p className="text-base font-semibold text-foreground">No one saved yet</p>
                <p className="mx-auto max-w-xs text-sm text-muted-foreground">Add the people you split with, with their WhatsApp number, and nudges go straight to their chat.</p>
              </>
            )}
          </div>
        )}

        {editing && (
          <PersonSheet
            key={editing === "new" ? "new" : editing.id}
            person={editing === "new" ? null : editing}
            balances={editing === "new" ? [] : balanceFor(editing.name)}
            recent={editing === "new" ? [] : recentFor(editing.name)}
            open
            onOpenChange={(o) => !o && setEditing(null)}
            onSave={(input) => (editing === "new" ? add(input.name, input.phone) : save(editing, input))}
            onDelete={
              editing === "new"
                ? undefined
                : () => {
                    toast.dismiss(); // a leftover toast would sit on top of the sheet's Delete button
                    setDeleting(editing);
                  }
            }
            onNudge={editing === "new" ? undefined : (b) => void nudge(editing, b)}
          />
        )}

        <ConfirmSheet
          open={!!deleting}
          onOpenChange={(o) => !o && setDeleting(null)}
          title={`Remove ${deleting?.name ?? "this person"}?`}
          description="They're removed from your people. Bills you've already split with them stay in History."
          confirmLabel="Remove"
          onConfirm={async () => {
            if (deleting) await remove(deleting);
          }}
        />
      </div>
    </PageTransition>
  );
}
