"use client";
import { useCallback, useEffect, useMemo, useState } from "react";
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
import type { Chip } from "@/lib/split/fallback-parser";
import { track, trackError } from "@/lib/telemetry/client";
import { primeOfflinePack } from "@/lib/ocr/prime";
import { bucketConfidence, bucketItems } from "@/lib/telemetry/events";
import { pinReceiptReader, releaseReceiptReader, scanReceipt, type ScanStage } from "@/lib/ocr/client";
import { useObjectUrl } from "@/components/receipt-photo";
import type { ClarifyAction } from "@/components/ai/ClarifyChips";
import type { Reading } from "@/components/ai/CrossCheck";
import type { SavedGroup } from "@/lib/db/schema";
import type { Cents, Receipt, ReceiptItem, SplitRecord } from "@/types/domain";

// All of the new-session flow's state and behaviour, in one place, so each step is a plain
// view over it. Money is integer cents the whole way through: the only floats are what a
// person types, and those go through parseMoneyInput() at the edge.

export type Step = "NAMES" | "SCAN" | "REVIEW" | "SUMMARY";
export const STEP_ORDER: Step[] = ["NAMES", "SCAN", "REVIEW", "SUMMARY"];

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

  const [step, setStep] = useState<Step>("NAMES");
  const [people, setPeople] = useState<string[]>([]);
  const [sessionName, setSessionName] = useState("");
  const [receipt, setReceipt] = useState<Receipt | null>(null);
  const [includeTax, setIncludeTax] = useState(true);
  const [instruction, setInstruction] = useState("");
  const [splitResult, setSplitResult] = useState("");
  const [structuredSplit, setStructuredSplit] = useState<SplitRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [user, setUser] = useState<CurrentUser>(null);
  const [isGuest, setIsGuest] = useState(false);
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

  // Group saving
  const [saveThisGroup, setSaveThisGroup] = useState(false);
  const [updateGroup, setUpdateGroup] = useState(true);
  const [groupName, setGroupName] = useState("");
  const [activeGroupId, setActiveGroupId] = useState<string | null>(null);
  const [originalPeople, setOriginalPeople] = useState<string[]>([]);
  const [savedGroups, setSavedGroups] = useState<SavedGroup[]>([]);
  // True while Start Scanning is saving the group: a second tap must not save it again.
  const [starting, setStarting] = useState(false);

  // ---- derived ----------------------------------------------------------------------
  const subtotal = receipt?.items.reduce((sum, item) => sum + item.totalPrice, 0) ?? 0;
  const displayedTotal = includeTax ? receipt?.total ?? 0 : subtotal;
  const symbol = receipt?.currency || "RM";

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

  // Identity comes from the verified session; groups are scoped to it on the server.
  useEffect(() => {
    (async () => {
      const u = await getCurrentUser();
      if (u) {
        setUser(u);
        setIsGuest(false);
        setSavedGroups(await listGroups().catch(() => []));
      } else {
        setIsGuest(true);
      }
    })();
  }, []);

  // Restore / start params
  useEffect(() => {
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
        if (bill.data.exactSplit && bill.data.cashRounding) {
          // Keep the exact shares so the rounding can still be changed or undone.
          setStructuredSplit(bill.data.exactSplit);
          setCashUnit(bill.data.cashRounding.unit);
          setAbsorberName(bill.data.cashRounding.absorber);
        } else {
          setStructuredSplit(bill.data.split);
        }
        setSplitResult(stripRoundingNote(bill.data.reasoning) || "Restored from history.");
        setSessionClientId(bill.clientId);
        setSessionName(bill.billTitle);
        setStep("SUMMARY");
      })();
      return;
    }

    if (namesParam) {
      setPeople(decodeURIComponent(namesParam).split(","));
    } else if (groupId) {
      (async () => {
        const data = await getGroup(groupId).catch(() => null);
        if (data) {
          setPeople(data.names);
          setGroupName(data.groupName);
          setActiveGroupId(data.id);
          setOriginalPeople(data.names);
        }
      })();
    }
  }, [searchParams, router]);

  // ---- people & groups --------------------------------------------------------------
  const hasGroupChanged = () => {
    if (!activeGroupId) return false;
    if (people.length !== originalPeople.length) return true;
    return JSON.stringify([...people].sort()) !== JSON.stringify([...originalPeople].sort());
  };

  const loadSavedGroup = (group: SavedGroup) => {
    setPeople(group.names);
    setGroupName(group.groupName);
    setActiveGroupId(group.id);
    setOriginalPeople(group.names);
  };

  /** Returns true if the name was added (so the input can clear). */
  const addPerson = (raw: string): boolean => {
    const name = raw.trim();
    if (!name || people.includes(name)) return false;
    setPeople((prev) => [...prev, name]);
    return true;
  };
  const removePerson = (name: string) => setPeople((prev) => prev.filter((p) => p !== name));

  const startScanning = async () => {
    if (starting) return; // a second tap while the first is still saving
    setStarting(true);
    try {
      if (user && people.length > 0) {
        // Saving a group is a nicety, never a reason to block scanning.
        try {
          if (activeGroupId && hasGroupChanged() && updateGroup) {
            await updateGroupNames(activeGroupId, people);
            setOriginalPeople(people); // it matches the saved group now
          } else if (!activeGroupId && saveThisGroup && groupName.trim()) {
            const saved = await saveGroup({ groupName, names: people });
            // Remember it. Otherwise Back -> Start Scanning would save the same group AGAIN
            // (the "Best Couple" x2 bug). The server also refuses a lookalike name, updating
            // the existing group instead, so we adopt whatever it returns.
            setActiveGroupId(saved.id);
            setGroupName(saved.groupName);
            setOriginalPeople(saved.names);
            setSavedGroups(await listGroups().catch(() => savedGroups));
          }
        } catch (e) {
          console.error("Could not save group", e);
        }
      }
      setStep("SCAN");
    } finally {
      setStarting(false);
    }
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
  const attemptSave = async (exact: SplitRecord[], log: string, unit = cashUnit, absorber = absorberName) => {
    if (!receipt) return;
    const rounded = roundSplit(exact, unit, absorber);
    // "Session N" needs the server's count; offline we fall back to a date.
    const finalTitle =
      sessionName.trim() || (await nextSessionTitle().catch(() => `Session ${new Date().toLocaleDateString()}`));
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
    await attemptSave(result.splits, result.reasoning);
    setStep("SUMMARY");
  };

  // The shares people actually pay: the engine's exact result, cash-rounded if asked.
  const rounded = useMemo(() => roundSplit(structuredSplit, cashUnit, absorberName), [structuredSplit, cashUnit, absorberName]);

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

  const split = () => runPlan([instruction]);

  /** A follow-up on the SUMMARY page, applied on top of everything said so far. */
  const modify = (text: string) => {
    const t = text.trim();
    if (t) void runPlan([...instructions, t]);
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
    router.push("/dashboard");
  }, [router]);

  return {
    // flow
    step, setStep, finish,
    // people & groups
    people, addPerson, removePerson, sessionName, setSessionName,
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
    disagree, pickReading, structuredSplit, splitResult,
    // cash rounding of the final shares
    finalSplit: rounded.splits, cashDeltas: rounded.deltas, absorberIndex: rounded.absorber,
    cashUnit, absorberName, chooseCashRounding, cashRoundingActive: rounded.changed,
    // AI keys
    keys, keySheetOpen, setKeySheetOpen,
  };
}

export type SplitSession = ReturnType<typeof useSplitSession>;
