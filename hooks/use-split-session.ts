"use client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { toast } from "sonner";

import { getCurrentUser } from "@/lib/actions/user";
import { getGroup, listGroups, saveGroup, updateGroupNames } from "@/lib/actions/groups";
import { getBill, nextSessionTitle } from "@/lib/actions/history";
import { enqueueBill } from "@/lib/sync/outbox";
import { inferMerchantCategory } from "@/lib/ai/merchantCategory";
import { planSplit, type PlanOutcome, type SplitChoice, type SplitPreview } from "@/lib/ai/planSplit";
import { explainAttempts } from "@/lib/ai/providers/cascade";
import { enhanceReceipt } from "@/lib/ai/providers/enhance";
import { ProviderError } from "@/lib/ai/providers/types";
import { getKeys, onKeysChanged, type Keys } from "@/lib/ai/keyStore";
import { formatMoney } from "@/lib/money";
import { roundSplit } from "@/lib/split/cashRound";
import { computeShareSplit } from "@/lib/split/shares";
import { percentOf } from "@/lib/split/sliders";
import type { Chip } from "@/lib/split/fallback-parser";
import { track, trackError } from "@/lib/telemetry/client";
import { primeOfflinePack } from "@/lib/ocr/prime";
import { bucketConfidence, bucketItems } from "@/lib/telemetry/events";
import { pinReceiptReader, releaseReceiptReader, scanReceipt, type ScanStage } from "@/lib/ocr/client";
import { useObjectUrl } from "@/components/receipt-photo";
import type { ClarifyAction } from "@/components/ai/ClarifyChips";
import type { Reading } from "@/components/ai/CrossCheck";
import type { SavedGroup } from "@/lib/db/schema";
import { defaultSettlement, guessMe, localDate, normalizeSettlement, settleView } from "@/lib/split/settle";
import { buildParticipants, contactLinks, knownLinks, meNameFor, nameProblem as checkName, resolveSelfRefs, samePerson } from "@/lib/split/participants";
import { listPeople } from "@/lib/actions/people";
import { usePhoneBook } from "@/hooks/use-phone-book";
import type { Cents, Receipt, ReceiptItem, Settlement, SplitRecord } from "@/types/domain";

// All of the new-session flow's state and behaviour, in one place, so each step is a plain
// view over it. Money is integer cents the whole way through: the only floats are what a
// person types, and those go through parseMoneyInput() at the edge.

// Who first, then how: NAMES -> (HOW) -> SCAN -> REVIEW -> SUMMARY, or NAMES -> (HOW) -> AMOUNT -> SPLIT -> SUMMARY.
// The older amount-first link (?mode=quick) keeps its order: AMOUNT -> NAMES -> SPLIT -> SUMMARY.
// Each order lists the steps so "forward" is simply a later index (used for the slide direction).
export type Step = "AMOUNT" | "NAMES" | "HOW" | "SCAN" | "REVIEW" | "SPLIT" | "SUMMARY";
export const STEP_ORDER: Step[] = ["AMOUNT", "NAMES", "SCAN", "REVIEW", "SPLIT", "SUMMARY"];
export const WHO_FIRST_ORDER: Step[] = ["NAMES", "HOW", "AMOUNT", "SCAN", "REVIEW", "SPLIT", "SUMMARY"];

/** How a who-first split gets its total: a scanned receipt or a typed amount. */
export type How = "scan" | "amount";

/** What the local scan was unsure about; drives the amber highlights on REVIEW. */
export interface ScanInfo {
  confidence: number;
  warnings: string[];
  itemConf: number[];
  empty: boolean;
  /** Only a total actually read off a TOTAL line is worth reconciling against. */
  printedTotal: Cents | null;
}

type Disagreement = { tier: "groq" | "gemini"; ai: SplitChoice; rules: SplitChoice; pending: string[] };
type Clarification = { chips: Chip[]; preview?: SplitPreview; pending: string[] };
type CurrentUser = Awaited<ReturnType<typeof getCurrentUser>>;

const escapeRe = (t: string) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

// The saved reasoning log gets the cash-rounding note appended after this marker; it is cut
// off again on restore so the note is never doubled up when the bill is re-saved.
const ROUNDING_NOTE_MARK = "\n\nCash rounding";
const stripRoundingNote = (reasoning: string) => reasoning.split(ROUNDING_NOTE_MARK)[0];

/** The receipt's total follows its items + tax + printed rounding, so an edit never leaves a stale total. */
export const withTotal = (r: Receipt): Receipt => ({
  ...r,
  total: r.items.reduce((sum, it) => sum + it.totalPrice, 0) + r.tax + (r.rounding ?? 0),
});

export function useSplitSession() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // ?mode=quick: split an amount without scanning (never when continuing a saved bill).
  const quick = searchParams.get("mode") === "quick" && !searchParams.get("restore");
  const [step, setStep] = useState<Step>(quick ? "AMOUNT" : "NAMES");
  // ?how=scan|amount|ask: who first, then how. Absent (an older link) = names then scan, as it always was.
  // "ask" shows the chooser after the names. A who-first split never asks to save a group up front; the
  // Summary offers it instead.
  const howParam = searchParams.get("how");
  const whoFirst = !quick && (howParam === "scan" || howParam === "amount" || howParam === "ask");
  const [how, setHow] = useState<How>(howParam === "amount" ? "amount" : "scan");
  const [people, setPeople] = useState<string[]>([]);
  // Which friends the people in this split stand for. The contacts decide for new names; a bill being edited keeps
  // what it recorded for the names it already had (`known`), so editing never shares a bill the owner left out.
  const { people: contacts } = usePhoneBook();
  const [known, setKnownState] = useState<Map<string, string | null> | null>(null);
  const knownRef = useRef<Map<string, string | null> | null>(null); // the same value, readable by a save that runs right after a change
  const setKnown = (next: Map<string, string | null> | null) => {
    knownRef.current = next;
    setKnownState(next);
  };
  const links = useMemo(() => contactLinks(contacts, known), [contacts, known]);
  const [sessionName, setSessionName] = useState("");
  const [receipt, setReceipt] = useState<Receipt | null>(null);
  const [includeTax, setIncludeTax] = useState(true);
  const [instruction, setInstruction] = useState("");
  const [splitResult, setSplitResult] = useState("");
  const [structuredSplit, setStructuredSplit] = useState<SplitRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [user, setUser] = useState<CurrentUser>(null);
  const [isGuest, setIsGuest] = useState(false);
  // Who the user is in THIS split. `me` is the name used for them in `people` (null = split for others).
  // `legacy` marks a restored bill saved before batch 5: it has no participants, so it keeps its guessed me
  // and is never rewritten with participants.
  const [userLoaded, setUserLoaded] = useState(false);
  const [me, setMe] = useState<string | null>(null);
  const [legacy, setLegacy] = useState(false);
  const startedRef = useRef(false);
  const [sessionClientId, setSessionClientId] = useState(() => crypto.randomUUID());
  const [scanStage, setScanStage] = useState<ScanStage | null>(null);
  // The conversation so far ("split equally", then "Aisha didn't have the rice"…): each
  // follow-up is applied on top of the earlier ones, later winning.
  const [instructions, setInstructions] = useState<string[]>([]);
  const [clarify, setClarify] = useState<Clarification | null>(null);
  const [splitNote, setSplitNote] = useState<string | null>(null);
  // The AI and the on-device rules disagreed: both readings wait here for the user's pick.
  const [disagree, setDisagree] = useState<Disagreement | null>(null);
  const [keys, setKeys] = useState<Keys>({});
  const [keySheetOpen, setKeySheetOpen] = useState(false);
  // The shrunk photo, kept in memory only so "Re-read with Gemini" can use it.
  const [photo, setPhoto] = useState<Blob | null>(null);
  const [enhancing, setEnhancing] = useState(false);
  const [enhanced, setEnhanced] = useState(false);
  const photoUrl = useObjectUrl(photo);
  const [showPhoto, setShowPhoto] = useState(false);
  const [scanInfo, setScanInfo] = useState<ScanInfo | null>(null);
  // Cash rounding of the final shares: 1 = exact. `absorberName` covers the difference (null = the biggest payer).
  const [cashUnit, setCashUnit] = useState(1);
  const [absorberName, setAbsorberName] = useState<string | null>(null);
  // Settle-up that has been SAVED with the bill. null until the user interacts with the card: just
  // looking at who owes whom must not turn a dinner everyone paid for in cash into an open debt.
  const [savedSettlement, setSavedSettlement] = useState<Settlement | null>(null);

  // Group saving
  const [saveThisGroup, setSaveThisGroup] = useState(false);
  const [updateGroup, setUpdateGroup] = useState(true);
  const [groupName, setGroupName] = useState("");
  const [activeGroupId, setActiveGroupId] = useState<string | null>(null);
  const [originalPeople, setOriginalPeople] = useState<string[]>([]);
  const [originalIncludesMe, setOriginalIncludesMe] = useState(true);
  const [savedGroups, setSavedGroups] = useState<SavedGroup[]>([]);
  // True while Start Scanning is saving the group: a second tap must not save it again.
  const [starting, setStarting] = useState(false);

  // ---- derived ----------------------------------------------------------------------
  const subtotal = receipt?.items.reduce((sum, item) => sum + item.totalPrice, 0) ?? 0;
  const displayedTotal = includeTax ? receipt?.total ?? 0 : subtotal;
  const symbol = receipt?.currency || "RM";
  // What the user is called in a split (first name, or "Me"), and whether they are in this one.
  const meName = meNameFor(user?.name);
  const hasMe = !!me && people.some((p) => samePerson(p, me));
  /** The group's people with the user put first (or left out), never listed twice. */
  const withMe = (names: string[], includeMe: boolean) => {
    const rest = names.filter((n) => !samePerson(n, meName));
    return includeMe ? [meName, ...rest] : rest;
  };

  // ---- lifecycle --------------------------------------------------------------------
  // Load the (self-hosted) reader while the user is still lining up the photo. While the scan
  // screen is open the reader stays warm and can't be freed under us.
  useEffect(() => {
    if (step === "SCAN") return pinReceiptReader();
  }, [step]);
  useEffect(() => () => void releaseReceiptReader(), []);
  // Once per browser, load the reader while online so the service worker can cache it.
  useEffect(() => {
    const idle = (window as Window & { requestIdleCallback?: (cb: () => void) => number }).requestIdleCallback;
    if (idle) idle(() => void primeOfflinePack());
    else setTimeout(() => void primeOfflinePack(), 2000);
  }, []);

  useEffect(() => {
    setKeys(getKeys());
    return onKeysChanged(() => setKeys(getKeys()));
  }, []);

  // Remember whether the user likes seeing the photo (read after mount: no hydration mismatch).
  useEffect(() => {
    try {
      setShowPhoto(localStorage.getItem("billa.review.showPhoto") === "1");
    } catch { /* storage blocked: default off */ }
  }, []);
  const togglePhoto = () =>
    setShowPhoto((v) => {
      try { localStorage.setItem("billa.review.showPhoto", v ? "0" : "1"); } catch { /* ignore */ }
      return !v;
    });

  // Identity comes from the verified session; groups are scoped to it on the server. Offline (or
  // if the call fails) we simply can't tell, so carry on as a guest: nothing here may block splitting.
  useEffect(() => {
    (async () => {
      const u = await getCurrentUser().catch(() => null);
      if (u) {
        setUser(u);
        setIsGuest(false);
        setUserLoaded(true);
        setSavedGroups(await listGroups().catch(() => []));
      } else {
        setIsGuest(true);
        setUserLoaded(true);
      }
    })();
  }, []);

  // Restore / start params. Runs once, after we know who the user is (their Me chip is named after them).
  useEffect(() => {
    if (!userLoaded || startedRef.current) return;
    startedRef.current = true;
    const restoreId = searchParams.get("restore");
    const namesParam = searchParams.get("names");
    const groupId = searchParams.get("group_id");

    // "Continue" from History: load the saved bill from the server (scoped to the signed-in
    // user), and keep its clientId so further edits UPDATE that history row instead of
    // creating a duplicate.
    if (restoreId) {
      (async () => {
        const bill = await getBill(restoreId).catch(() => null);
        if (!bill) {
          router.replace("/dashboard/history");
          return;
        }
        setReceipt(bill.data.items);
        setPeople(bill.data.people);
        // A bill saved with participants knows who the user is (or that it was split for others); an older one
        // only has a guess, so it keeps the old behaviour.
        setLegacy(!bill.data.participants);
        setKnown(knownLinks(bill.data));
        setMe(bill.data.participants?.find((p) => p.me)?.name ?? null);
        if (bill.data.exactSplit && bill.data.cashRounding) {
          // Keep the exact shares so the rounding can still be changed or undone.
          setStructuredSplit(bill.data.exactSplit);
          setCashUnit(bill.data.cashRounding.unit);
          setAbsorberName(bill.data.cashRounding.absorber);
        } else {
          setStructuredSplit(bill.data.split);
        }
        setSavedSettlement(bill.data.settlement ?? null);
        setSplitResult(stripRoundingNote(bill.data.reasoning) || "Restored from history.");
        setSessionClientId(bill.clientId);
        setSessionName(bill.billTitle);
        setStep("SUMMARY");
      })();
      return;
    }

    if (namesParam) {
      setPeople(withMe(decodeURIComponent(namesParam).split(","), true));
      setMe(meName);
    } else if (groupId) {
      (async () => {
        const data = await getGroup(groupId).catch(() => null);
        if (data) {
          setPeople(withMe(data.names, data.includesMe));
          setMe(data.includesMe ? meName : null);
          setGroupName(data.groupName);
          setActiveGroupId(data.id);
          setOriginalPeople(data.names);
          setOriginalIncludesMe(data.includesMe);
        } else {
          setPeople([meName]);
          setMe(meName);
        }
      })();
    } else {
      // A fresh split starts with you in it.
      setPeople([meName]);
      setMe(meName);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- runs once, when the user becomes known
  }, [userLoaded, searchParams, router, meName]);

  // ---- people & groups --------------------------------------------------------------
  // A saved group holds the people OTHER than the user, plus whether the user is in it.
  const groupNames = people.filter((p) => !(me && samePerson(p, me)));
  const hasGroupChanged = () => {
    if (!activeGroupId) return false;
    if (hasMe !== originalIncludesMe) return true;
    if (groupNames.length !== originalPeople.length) return true;
    return JSON.stringify([...groupNames].sort()) !== JSON.stringify([...originalPeople].sort());
  };

  const loadSavedGroup = (group: SavedGroup) => {
    setPeople(withMe(group.names, group.includesMe));
    setMe(group.includesMe ? meName : null);
    setGroupName(group.groupName);
    setActiveGroupId(group.id);
    setOriginalPeople(group.names);
    setOriginalIncludesMe(group.includesMe);
  };

  /** Why this name can't be added (shown under the input), or null. Case and spacing don't make a new person. */
  const nameProblem = (raw: string) => checkName(raw, people, hasMe ? me : null);
  /** Returns true if the name was added (so the input can clear). */
  const addPerson = (raw: string): boolean => {
    if (nameProblem(raw)) return false;
    const name = raw.trim();
    setPeople((prev) => [...prev, name]);
    // Typing your own name while you're out of the split is the same as putting yourself back in.
    if (!legacy && !me && samePerson(name, meName)) setMe(name);
    return true;
  };
  const removePerson = (name: string) => {
    setPeople((prev) => prev.filter((p) => p !== name));
    if (me && samePerson(name, me)) setMe(null);
  };
  /** Put yourself back in after removing the Me chip. */
  const addMe = () => {
    if (hasMe) return;
    setPeople((prev) => [meName, ...prev.filter((p) => !samePerson(p, meName))]);
    setMe(meName);
  };

  const startScanning = async () => {
    if (starting) return; // a second tap while the first is still saving
    setStarting(true);
    try {
      if (user && people.length > 0) {
        // Saving a group is a nicety, never a reason to block scanning.
        try {
          if (activeGroupId && hasGroupChanged() && updateGroup) {
            await updateGroupNames(activeGroupId, groupNames, hasMe);
            setOriginalPeople(groupNames); // it matches the saved group now
            setOriginalIncludesMe(hasMe);
          } else if (!activeGroupId && saveThisGroup && groupName.trim()) {
            const saved = await saveGroup({ groupName, names: groupNames, includesMe: hasMe });
            // Remember it. Otherwise Back -> Start Scanning would save the same group AGAIN
            // (the "Best Couple" x2 bug). The server also refuses a lookalike name, updating
            // the existing group instead, so we adopt whatever it returns.
            setActiveGroupId(saved.id);
            setGroupName(saved.groupName);
            setOriginalPeople(saved.names);
            setOriginalIncludesMe(saved.includesMe);
            setSavedGroups(await listGroups().catch(() => savedGroups));
          }
        } catch (e) {
          console.error("Could not save group", e);
        }
      }
      setStep(quick ? "SPLIT" : howParam === "ask" ? "HOW" : how === "amount" ? "AMOUNT" : "SCAN");
    } finally {
      setStarting(false);
    }
  };

  // ---- save the people as a group, after the fact (offered on the Summary) ----------------
  /** Signed in, with at least two people, and not already from a saved group. */
  const canSaveAsGroup = !isGuest && !activeGroupId && people.length >= 2 && !legacy;
  /** Resolves to an error message, or null when saved. */
  const saveAsGroup = async (name: string): Promise<string | null> => {
    const trimmed = name.trim();
    if (!trimmed) return "Give the group a name.";
    try {
      const saved = await saveGroup({ groupName: trimmed, names: groupNames, includesMe: hasMe });
      setActiveGroupId(saved.id);
      setGroupName(saved.groupName);
      setOriginalPeople(saved.names);
      setOriginalIncludesMe(saved.includesMe);
      setSavedGroups(await listGroups().catch(() => savedGroups));
      return null;
    } catch {
      return "Couldn't save the group. Check your connection and try again.";
    }
  };

  // ---- quick split: an amount instead of a scanned receipt ----------------------------
  /** Builds a one-item receipt from the typed amount (and optional tax) and moves on to the people. */
  const startQuick = ({ cents, title, taxCents }: { cents: Cents; title: string; taxCents: Cents }) => {
    const name = title.trim();
    setReceipt(
      withTotal({
        items: [{ name: name || "Quick split", quantity: 1, unitPrice: cents, totalPrice: cents }],
        tax: taxCents,
        total: 0,
        currency: "RM",
      }),
    );
    if (name) setSessionName(name);
    setIncludeTax(true);
    setStep(quick ? "NAMES" : "SPLIT");
  };

  /** The chooser after the names: scan a receipt or type an amount. */
  const chooseHow = (next: How) => {
    setHow(next);
    setStep(next === "amount" ? "AMOUNT" : "SCAN");
  };

  /** One step back. Quick split walks its own path; the scan flow has always gone back to the names. */
  const back = () => {
    if (quick) {
      const prev: Partial<Record<Step, Step>> = { NAMES: "AMOUNT", SPLIT: "NAMES", SUMMARY: "SPLIT" };
      return setStep(prev[step] ?? "AMOUNT");
    }
    const ask = howParam === "ask";
    if (how === "amount") {
      // Typing an amount: the chooser (when it was shown) sits between the names and the amount.
      const prev: Partial<Record<Step, Step>> = { AMOUNT: ask ? "HOW" : "NAMES", SPLIT: "AMOUNT", SUMMARY: "SPLIT" };
      return setStep(prev[step] ?? "NAMES");
    }
    if (step === "SCAN" && ask) return setStep("HOW");
    setStep("NAMES");
  };

  // ---- scanning ---------------------------------------------------------------------
  // Scanning happens entirely on this device: shrink the photo, read it with the local OCR
  // engine, parse it. Nothing is uploaded, so it also works offline.
  /** Returns false if the photo was rejected or unreadable (so the file input can be reset). */
  const scanFile = async (file: File): Promise<boolean> => {
    if (!file.type.startsWith("image/")) {
      toast.error("Please choose an image file (JPG or PNG).");
      return false;
    }
    setLoading(true);
    const scanStarted = performance.now();
    try {
      const { parsed, prepared } = await scanReceipt(file, setScanStage);
      track({
        type: "scan",
        engine: "tesseract",
        confidence: bucketConfidence(parsed.confidence),
        items: bucketItems(parsed.items.length),
        totalSource: parsed.totalSource,
        warnings: Math.min(parsed.warnings.length, 20),
        ms: Math.round(performance.now() - scanStarted),
      });
      setPhoto(prepared.display);
      setEnhanced(false);
      setInstructions([]);
      setClarify(null);
      setReceipt(parsed.receipt);
      setScanInfo({
        confidence: parsed.confidence,
        warnings: parsed.warnings,
        itemConf: parsed.items.map((it) => it.confidence),
        empty: parsed.items.length === 0,
        printedTotal: parsed.totalSource === "keyword" ? parsed.receipt.total : null,
      });
      // Even a failed read goes to REVIEW: the user can type the items in, which is better
      // than a dead end.
      setStep("REVIEW");
      return true;
    } catch (err) {
      console.error("Scan Error:", err);
      trackError("scan", err);
      const msg = err instanceof Error ? err.message : "";
      toast.error(msg.startsWith("Couldn't read") ? msg : "Could not scan that photo. Try again, or add the items by hand.");
      return false;
    } finally {
      setScanStage(null);
      setLoading(false);
    }
  };

  // ---- REVIEW edits (all in cents) --------------------------------------------------
  const clearConf = (i: number) =>
    setScanInfo((si) => (si ? { ...si, itemConf: si.itemConf.map((c, k) => (k === i ? 1 : c)) } : si));
  const editItem = (i: number, patch: Partial<ReceiptItem>) => {
    setReceipt((prev) => {
      if (!prev) return prev;
      const items = prev.items.map((it, k) => {
        if (k !== i) return it;
        const next = { ...it, ...patch };
        next.unitPrice = next.quantity > 0 ? Math.round(next.totalPrice / next.quantity) : next.totalPrice;
        return next;
      });
      return withTotal({ ...prev, items });
    });
    clearConf(i);
  };
  const removeItem = (i: number) => {
    setReceipt((prev) => (prev ? withTotal({ ...prev, items: prev.items.filter((_, k) => k !== i) }) : prev));
    setScanInfo((si) => (si ? { ...si, itemConf: si.itemConf.filter((_, k) => k !== i) } : si));
  };
  const addItem = () => {
    setReceipt((prev) =>
      prev ? { ...prev, items: [...prev.items, { name: "", quantity: 1, unitPrice: 0, totalPrice: 0 }] } : prev,
    );
    setScanInfo((si) => (si ? { ...si, itemConf: [...si.itemConf, 1] } : si));
  };
  const editTax = (tax: Cents) => setReceipt((prev) => (prev ? withTotal({ ...prev, tax }) : prev));
  /** Signed cash-rounding adjustment from the receipt; 0 removes it. */
  const editRounding = (rounding: Cents) =>
    setReceipt((prev) => {
      if (!prev) return prev;
      const next: Receipt = { ...prev, rounding };
      if (rounding === 0) delete next.rounding;
      return withTotal(next);
    });

  // ---- splitting: route -> ambiguity check -> Groq -> Gemini -> on-device rules -> engine
  const describeTier = (o: Extract<PlanOutcome, { kind: "split" }>) => {
    const why = explainAttempts(o.attempts);
    if (o.tier === "groq") return "Split by Groq using your key.";
    if (o.tier === "gemini") return why ? `${why}, so Gemini split this.` : "Split by Gemini using your key.";
    return why
      ? `${why} — used on-device rules instead.`
      : "Split with on-device rules. Add a free Groq key for smarter splits.";
  };

  /** The plain-words note for a cash rounding: who moved by how much, and who covers the difference. */
  const roundingNote = (rounded: ReturnType<typeof roundSplit>, unit: number) => {
    const moves = rounded.splits
      .map((s, i) => ({ name: s.name, d: rounded.deltas[i] }))
      .filter((m) => m.d !== 0)
      .map((m) => `${m.name} ${m.d > 0 ? "+" : "-"}${formatMoney(Math.abs(m.d), symbol)}`);
    return `${ROUNDING_NOTE_MARK} to ${formatMoney(unit, symbol)}: ${moves.join(", ")}. ${rounded.splits[rounded.absorber].name} covers the difference so the total stays exact.`;
  };

  /** Saves the bill (exact shares in, cash rounding applied on top). `exact` is the engine's result. */
  const attemptSave = async (
    exact: SplitRecord[],
    log: string,
    unit = cashUnit,
    absorber = absorberName,
    settle: Settlement | null = savedSettlement,
  ) => {
    if (!receipt) return;
    const rounded = roundSplit(exact, unit, absorber);
    // "Session N" needs the server's count; offline we fall back to a date.
    const finalTitle =
      sessionName.trim() || (await nextSessionTitle().catch(() => `Session ${new Date().toLocaleDateString()}`));
    // Who was in it and which one is the user. settlement.me stays written for older readers, so it must agree.
    // Contacts that haven't loaded yet (a fast save) are fetched now, so a friend is never saved as a plain name by accident.
    const linkMap = contactLinks(contacts ?? (await listPeople().catch(() => null)), knownRef.current);
    const participants = legacy ? null : buildParticipants(people, hasMe ? me : null, linkMap);
    const meInBill = participants?.find((p) => p.me)?.name;
    const settleOut = settle && participants ? { ...settle, me: meInBill } : settle;
    try {
      await enqueueBill(user?.id ?? null, {
        clientId: sessionClientId,
        billTitle: finalTitle,
        // No merchant name is scanned yet, so this is inferred from the items.
        merchantCategory: inferMerchantCategory(null, receipt.items),
        totalAmount: displayedTotal,
        currency: symbol,
        data: {
          split: rounded.splits,
          items: receipt,
          people,
          reasoning: rounded.changed ? log + roundingNote(rounded, unit) : log,
          ...(rounded.changed ? { exactSplit: exact, cashRounding: { unit, absorber: exact[rounded.absorber].name } } : {}),
          ...(participants ? { participants } : {}),
          ...(settleOut ? { settlement: settleOut } : {}),
        },
      });
    } catch (e) {
      console.error("Save error:", e);
      toast.error("Couldn't save this session on your device. Please try again.");
    }
  };

  const applySplit = async (result: SplitPreview["result"], note: string, next: string[]) => {
    setStructuredSplit(result.splits);
    setSplitResult(result.reasoning);
    setInstructions(next);
    setSplitNote(note);
    setClarify(null);
    setDisagree(null);
    // A saved settlement follows the new shares (its single payer fronted the whole new total).
    const settle = savedSettlement ? normalizeSettlement(savedSettlement, result.splits) : null;
    setSavedSettlement(settle);
    await attemptSave(result.splits, result.reasoning, cashUnit, absorberName, settle);
    setStep("SUMMARY");
  };

  // The shares people actually pay: the engine's exact result, cash-rounded if asked.
  const rounded = useMemo(() => roundSplit(structuredSplit, cashUnit, absorberName), [structuredSplit, cashUnit, absorberName]);

  // ---- settle-up: who paid, who owes whom ---------------------------------------------------
  // The settlement shown is the saved one, or (until the user touches the card) a default: the user,
  // if we can tell who that is, fronted it. Transfers are derived from the shares people actually pay.
  const settlement = useMemo(() => {
    if (rounded.splits.length === 0) return null;
    // New bills know who the user is; a restored older bill keeps its first-name guess.
    const meForSettle = legacy ? guessMe(rounded.splits.map((s) => s.name), user?.name) : hasMe && me ? me : undefined;
    const base = savedSettlement ?? defaultSettlement(rounded.splits, meForSettle);
    return normalizeSettlement(legacy ? base : { ...base, me: meForSettle }, rounded.splits);
  }, [savedSettlement, rounded.splits, user?.name, legacy, hasMe, me]);

  /** Change the settlement and save it with the bill. The first change is what starts tracking it. */
  const changeSettlementNow = async (change: (s: Settlement) => Settlement) => {
    if (!settlement) return;
    let next = change(settlement);
    const view = settleView(rounded.splits, next, localDate());
    if (view && view.ok) next = view.settlement; // drop paid marks that no longer match
    setSavedSettlement(next);
    await attemptSave(structuredSplit, splitResult, cashUnit, absorberName, next);
  };

  // The identity of this function never changes and it always runs the latest version above, so an
  // Undo toast created a minute ago can't act on (and overwrite) a stale copy of the settlement.
  const changeRef = useRef(changeSettlementNow);
  useEffect(() => {
    changeRef.current = changeSettlementNow;
  });
  const changeSettlement = useCallback((change: (s: Settlement) => Settlement) => changeRef.current(change), []);

  /** Share the bill with a friend on it (their account), or stop sharing it (null). The person stays on the bill. */
  const setShared = async (name: string, userId: string | null) => {
    const next = new Map(knownRef.current ?? []);
    next.set(name.trim().toLowerCase(), userId);
    setKnown(next);
    await attemptSave(structuredSplit, splitResult, cashUnit, absorberName, savedSettlement);
  };

  const stopTracking = async () => {
    setSavedSettlement(null);
    await attemptSave(structuredSplit, splitResult, cashUnit, absorberName, null);
  };

  /** Change the cash rounding (1 = exact) and, optionally, who covers the difference; re-saves the bill. */
  const chooseCashRounding = async (unit: number, absorber: string | null = absorberName) => {
    setCashUnit(unit);
    setAbsorberName(absorber);
    await attemptSave(structuredSplit, splitResult, unit, absorber);
  };

  const runPlan = async (next: string[], opts: { ignoreAmbiguities?: boolean; peopleOverride?: string[] } = {}) => {
    if (!receipt) return;
    setLoading(true);
    setClarify(null);
    setDisagree(null);
    const splitStarted = performance.now();
    try {
      const outcome = await planSplit({
        receipt,
        people: opts.peopleOverride ?? people,
        instructions: next,
        applyTax: includeTax,
        keys: getKeys(),
        ignoreAmbiguities: opts.ignoreAmbiguities,
      });

      if (outcome.kind === "add-members") {
        const fresh = outcome.names.filter((n) => !people.some((p) => p.toLowerCase() === n.toLowerCase()));
        const nextPeople = [...people, ...fresh];
        setPeople(nextPeople);
        setInstruction("");
        if (step === "SUMMARY") {
          // Someone joined after the split: recompute so they get their share.
          await runPlan(instructions, { peopleOverride: nextPeople });
        } else {
          setSplitNote(`Added ${fresh.join(", ") || "nobody new"}. Now tell me how to split.`);
        }
      } else if (outcome.kind === "shares") {
        await applySplit(outcome.result, outcome.summary, next);
      } else if (outcome.kind === "needs-clarification") {
        setClarify({ chips: outcome.chips, preview: outcome.preview, pending: next });
      } else if (outcome.kind === "disagreement") {
        track({ type: "crosscheck", tier: outcome.ai.tier, outcome: "shown" });
        setDisagree({ tier: outcome.ai.tier, ai: outcome.ai, rules: outcome.rules, pending: next });
      } else {
        track({
          type: "split",
          tier: outcome.tier,
          attempts: outcome.attempts.filter((a) => !a.ok).slice(0, 6).map((a) => ({ tier: a.tier, kind: a.kind ?? "network" })),
          repairs: Math.min(outcome.repairs.length, 50),
          ms: Math.round(performance.now() - splitStarted),
          rounds: Math.min(Math.max(next.length, 1), 50),
        });
        await applySplit(outcome.result, describeTier(outcome), next);
      }
    } catch (err) {
      console.error("Split error:", err);
      trackError("split", err);
      toast.error("Something went wrong splitting this. Try rewording, or check the items.");
    } finally {
      setLoading(false);
    }
  };

  const split = () => runPlan([resolveSelfRefs(instruction, hasMe ? me : null, people)]);

  /** A follow-up on the SUMMARY page, applied on top of everything said so far. */
  const modify = (text: string) => {
    const t = text.trim();
    if (t) void runPlan([...instructions, resolveSelfRefs(t, hasMe ? me : null, people)]);
  };

  /**
   * Shares set directly (the sliders, or Equal): cents per person, in `people` order, adding up to
   * the bill. Pure arithmetic on the device, same path as a typed "Alexis 70%, Riley 30%".
   */
  const applyShares = async (amounts: Cents[]) => {
    if (!receipt || amounts.length !== people.length) return;
    const total = amounts.reduce((a, b) => a + b, 0); // what is being shared: the bill, or what the split covers
    const pct = (a: Cents) => `${percentOf(a, total)}%`;
    try {
      const result = computeShareSplit(
        people,
        amounts,
        amounts.map((a) => `${pct(a)} of the bill`),
        total,
        [`Shares set with the sliders: ${people.map((p, i) => `${p} ${pct(amounts[i])} (${formatMoney(amounts[i], symbol)})`).join(", ")}.`],
      );
      await applySplit(result, "Split by the shares you set, worked out on your device.", []);
    } catch (err) {
      console.error("Share split error:", err);
      toast.error("Those shares don't add up to the bill. Reset them and try again.");
    }
  };

  const pickReading = async (which: Reading) => {
    if (!disagree) return;
    const { tier, ai, rules, pending } = disagree;
    track({ type: "crosscheck", tier, outcome: which });
    const name = tier === "groq" ? "Groq" : "Gemini";
    await applySplit(
      (which === "ai" ? ai : rules).result,
      which === "ai" ? `Split by ${name} — you chose the AI's reading.` : "Split with the built-in rules — you chose their reading.",
      pending,
    );
  };

  const handleClarify = async (a: ClarifyAction) => {
    if (!clarify) return;
    const pending = clarify.pending;
    if (a.type === "edit") return setClarify(null);
    if (a.type === "use-preview") {
      if (clarify.preview) await applySplit(clarify.preview.result, "Split with on-device rules; the unclear parts were shared equally.", pending);
      return;
    }
    if (a.type === "continue-anyway") return runPlan(pending, { ignoreAmbiguities: true });
    if (a.type === "add-person") {
      const nextPeople = people.some((p) => p.toLowerCase() === a.name.toLowerCase()) ? people : [...people, a.name];
      setPeople(nextPeople);
      return runPlan(pending, { peopleOverride: nextPeople });
    }
    if (a.type === "append") {
      // A one-tap fix from a shares question: add the clause to the latest instruction and re-run.
      const fixed = pending.map((t, i) => (i === pending.length - 1 ? `${t}, ${a.text}` : t));
      if (fixed.length === 1) setInstruction(fixed[0]);
      return runPlan(fixed);
    }
    // replace: fix the typo / pick the item, then re-run
    const re = new RegExp(`\\b${escapeRe(a.from)}\\b`, "gi");
    const fixed = pending.map((t) => t.replace(re, a.to));
    if (fixed.length === 1) setInstruction(fixed[0]);
    return runPlan(fixed);
  };

  // ---- Cloud Enhance: opt-in, per scan, with the user's own Gemini key ----------------
  const enhance = async () => {
    const key = keys.gemini;
    if (!key || !photo) return;
    setEnhancing(true);
    const enhanceStarted = performance.now();
    try {
      const parsed = await enhanceReceipt(photo, key, { signal: AbortSignal.timeout(45_000) }); // vision on the free tier measured ~15 s
      setReceipt(parsed.receipt);
      setScanInfo({
        confidence: parsed.confidence,
        warnings: parsed.warnings,
        itemConf: parsed.items.map((it) => it.confidence),
        empty: parsed.items.length === 0,
        printedTotal: parsed.totalSource === "keyword" ? parsed.receipt.total : null,
      });
      setEnhanced(true);
      track({ type: "enhance", ok: true, ms: Math.round(performance.now() - enhanceStarted) });
    } catch (err) {
      const kind = err instanceof ProviderError ? err.kind : "network";
      track({ type: "enhance", ok: false, kind, ms: Math.round(performance.now() - enhanceStarted) });
      toast.error(
        kind === "auth" ? "Gemini rejected your key. Check it in AI settings."
        : kind === "rate-limit" ? "Gemini is rate-limited right now. Try again in a minute."
        : kind === "overloaded" ? "Gemini is overloaded right now — that's on Google's side, not your key. Try again in a minute."
        : kind === "blocked" ? "Gemini declined to read this photo."
        : "Couldn't get a second reading from Gemini. You can still fix the items by hand.",
      );
    } finally {
      setEnhancing(false);
    }
  };

  const finish = useCallback(() => {
    router.refresh();
    router.push("/dashboard", { transitionTypes: ["nav-back"] });
  }, [router]);

  return {
    // flow
    step, setStep, finish, quick, back, startQuick, firstStep: (quick ? "AMOUNT" : "NAMES") as Step,
    // who first, then how
    canSaveAsGroup, saveAsGroup, whoFirst, how, chooseHow, stepOrder: quick ? STEP_ORDER : WHO_FIRST_ORDER, amountFlow: quick || how === "amount",
    // people & groups
    people, addPerson, removePerson, nameProblem, me, meName, hasMe, addMe, legacy, contacts, setShared,
    // who was in it (null for an older bill): handed to Settle up, which then stops asking "which one is you?"
    participants: legacy ? null : buildParticipants(people, hasMe ? me : null, links),
    hosted: !legacy && !hasMe,
    sessionName, setSessionName,
    isGuest, savedGroups, loadSavedGroup, activeGroupId, groupName, setGroupName,
    saveThisGroup, setSaveThisGroup, updateGroup, setUpdateGroup, hasGroupChanged,
    starting, startScanning,
    // scan
    loading, scanStage, scanFile,
    photo, photoUrl, showPhoto, togglePhoto, enhance, enhancing, enhanced, scanInfo,
    // review
    receipt, symbol, subtotal, displayedTotal, includeTax, setIncludeTax,
    editItem, removeItem, addItem, editTax, editRounding,
    // split
    instruction, setInstruction, split, modify, splitNote, clarify, handleClarify,
    disagree, pickReading, applyShares, structuredSplit, splitResult,
    // cash rounding of the final shares
    finalSplit: rounded.splits, cashDeltas: rounded.deltas, absorberIndex: rounded.absorber,
    cashUnit, absorberName, chooseCashRounding, cashRoundingActive: rounded.changed,
    // settle-up
    settlement, settleTracked: savedSettlement !== null, changeSettlement, stopTracking,
    billTitle: sessionName.trim() || "the bill",
    // AI keys
    keys, keySheetOpen, setKeySheetOpen,
  };
}

export type SplitSession = ReturnType<typeof useSplitSession>;
