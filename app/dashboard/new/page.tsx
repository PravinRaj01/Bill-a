"use client";
import { useState, useRef, useEffect, Suspense } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import {
  Table,
  TableBody,
  TableCell,
  TableRow,
} from "@/components/ui/table";
import { getCurrentUser } from "@/lib/actions/user";
import { getGroup, listGroups, saveGroup, updateGroupNames } from "@/lib/actions/groups";
import { getBill, nextSessionTitle } from "@/lib/actions/history";
import { enqueueBill } from "@/lib/sync/outbox";
import { inferMerchantCategory } from "@/lib/ai/merchantCategory";
import { receiptToDomain, receiptToLegacy, splitsToDomain, splitsToLegacy, toCents } from "@/lib/money";
import { planSplit, type PlanOutcome, type SplitPreview } from "@/lib/ai/planSplit";
import { explainAttempts } from "@/lib/ai/providers/cascade";
import { enhanceReceipt } from "@/lib/ai/providers/enhance";
import { ProviderError } from "@/lib/ai/providers/types";
import { getKeys, onKeysChanged, type Keys } from "@/lib/ai/keyStore";
import type { Chip } from "@/lib/split/fallback-parser";
import { ApiKeySettings } from "@/components/ai/ApiKeySettings";
import { ClarifyChips, type ClarifyAction } from "@/components/ai/ClarifyChips";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { releaseReceiptReader, scanReceipt, warmReceiptReader, type ScanStage } from "@/lib/ocr/client";
import { useRouter, useSearchParams } from "next/navigation";
import {
  Loader2,
  Plus,
  Trash2,
  TriangleAlert,
  Receipt,
  Share2,
  X,
  ChevronLeft,
  Image as ImageIcon,
  ScrollText,
  ChevronUp,
  ChevronDown,
  User,
  Users,
  RefreshCw,
  MessageSquare, 
  Sparkles      
} from "lucide-react";
import { Label } from "@/components/ui/label";

interface ReceiptItem { name: string; total_price: number; quantity: number; unit_price: number; }
interface ReceiptData { items: ReceiptItem[]; tax: number; total: number; currency: string; }
interface SplitRecord { name: string; amount: number; items: string; }
type Step = "NAMES" | "SCAN" | "REVIEW" | "SUMMARY";

function BillSplitterContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const [step, setStep] = useState<Step>("NAMES");
  const [people, setPeople] = useState<string[]>([]);
  const [newName, setNewName] = useState("");
  const [sessionName, setSessionName] = useState("");
  const [items, setItems] = useState<ReceiptData | null>(null);
  const [includeTax, setIncludeTax] = useState(true);
  const [instruction, setInstruction] = useState("");
  const [splitResult, setSplitResult] = useState("");
  const [structuredSplit, setStructuredSplit] = useState<SplitRecord[]>([]);
  const [showReasoning, setShowReasoning] = useState(false);
  const [loading, setLoading] = useState(false);
  const [user, setUser] = useState<any>(null);
  const [isGuest, setIsGuest] = useState(false);
  const [sessionClientId, setSessionClientId] = useState(() => crypto.randomUUID());
  // What the local scan was unsure about; drives the amber highlights on REVIEW.
  const [scanStage, setScanStage] = useState<ScanStage | null>(null);
  // The conversation so far ("split equally", then "Aisha didn't have the rice"…): each
  // follow-up is applied on top of the earlier ones, later winning.
  const [instructions, setInstructions] = useState<string[]>([]);
  const [modifyText, setModifyText] = useState("");
  const [clarify, setClarify] = useState<{ chips: Chip[]; preview?: SplitPreview; pending: string[] } | null>(null);
  const [splitNote, setSplitNote] = useState<string | null>(null);
  const [keys, setKeys] = useState<Keys>({});
  const [keySheetOpen, setKeySheetOpen] = useState(false);
  // The shrunk photo, kept in memory only so "Re-read with Gemini" can use it.
  const [photo, setPhoto] = useState<Blob | null>(null);
  const [enhancing, setEnhancing] = useState(false);
  const [enhanced, setEnhanced] = useState(false);
  const [scanInfo, setScanInfo] = useState<{ confidence: number; warnings: string[]; itemConf: number[]; empty: boolean; printedTotal: number | null } | null>(null);

  // GROUP SAVING STATE
  const [saveThisGroup, setSaveThisGroup] = useState(false);
  const [updateGroup, setUpdateGroup] = useState(true); 
  const [groupName, setGroupName] = useState("");
  const [activeGroupId, setActiveGroupId] = useState<string | null>(null);
  const [originalPeople, setOriginalPeople] = useState<string[]>([]); 
  
  // GROUP LOADING UI STATE
  const [savedGroups, setSavedGroups] = useState<any[]>([]);
  const [showGroupList, setShowGroupList] = useState(false);

  // Load the (self-hosted) reader while the user is still lining up the photo.
  useEffect(() => {
    if (step === "SCAN") warmReceiptReader().catch(() => {});
  }, [step]);
  useEffect(() => () => void releaseReceiptReader(), []);

  useEffect(() => {
    setKeys(getKeys());
    return onKeysChanged(() => setKeys(getKeys()));
  }, []);

  const isCreator = true;
  const fileInputRef = useRef<HTMLInputElement>(null);
  const galleryRef = useRef<HTMLInputElement>(null);

  const subtotal = items?.items.reduce((sum, item) => sum + item.total_price, 0) || 0;
  const displayedTotal = includeTax ? items?.total || 0 : subtotal;

  // 1. Check User & Fetch All Saved Groups
  useEffect(() => {
    const init = async () => {
      // Identity comes from the verified session; groups are scoped to it on the server.
      const u = await getCurrentUser();
      if (u) { 
        setUser(u); 
        setIsGuest(false);
        setSavedGroups(await listGroups().catch(() => []));
      } else { 
        setIsGuest(true); 
      }
    };
    init();
  }, []);

  // 2. RESTORE / START PARAMS
  useEffect(() => {
    const restoreId = searchParams.get("restore");
    const namesParam = searchParams.get("names");
    const groupId = searchParams.get("group_id");

    // A. "Continue" from History: load the saved bill from the server (scoped to the
    //    signed-in user), and keep its clientId so further edits UPDATE that history
    //    row instead of creating a duplicate.
    if (restoreId) {
        const load = async () => {
            const bill = await getBill(restoreId).catch(() => null);
            if (!bill) {
                router.replace("/dashboard/history");
                return;
            }
            setItems(receiptToLegacy(bill.data.items));
            setPeople(bill.data.people);
            setStructuredSplit(splitsToLegacy(bill.data.split));
            setSplitResult(bill.data.reasoning || "Restored from history.");
            setSessionClientId(bill.clientId);
            setSessionName(bill.billTitle);
            setStep("SUMMARY");
        };
        load();
        return;
    }

    // B. NEW SESSION (URL params)
    if (namesParam) {
        setPeople(decodeURIComponent(namesParam).split(","));
    } else if (groupId) {
        const loadGroup = async () => {
            const data = await getGroup(groupId).catch(() => null);
            if (data) {
                setPeople(data.names);
                setGroupName(data.groupName);
                setActiveGroupId(data.id);
                setOriginalPeople(data.names);
            }
        };
        loadGroup();
    }
  }, [searchParams, router]);

  const hasGroupChanged = () => {
      if (!activeGroupId) return false;
      if (people.length !== originalPeople.length) return true;
      const sortedPeople = [...people].sort();
      const sortedOriginal = [...originalPeople].sort();
      return JSON.stringify(sortedPeople) !== JSON.stringify(sortedOriginal);
  };

  const loadSavedGroup = (group: any) => {
      setPeople(group.names);
      setGroupName(group.groupName);
      setActiveGroupId(group.id);
      setOriginalPeople(group.names);
      setShowGroupList(false); 
  };

  const symbol = items?.currency || "RM";

  const addPerson = () => {
    const name = newName.trim();
    if (name && !people.includes(name)) {
      setPeople((prev) => [...prev, name]);
      setNewName("");
    }
  };

  const removePerson = (nameToRemove: string) => {
    setPeople((prev) => prev.filter((p) => p !== nameToRemove));
  };

  const handleStartScanning = async () => {
    if (user && people.length > 0) {
        // Saving a group is a nicety, never a reason to block scanning.
        try {
            if (activeGroupId && hasGroupChanged() && updateGroup) {
                await updateGroupNames(activeGroupId, people);
            } else if (!activeGroupId && saveThisGroup && groupName) {
                await saveGroup({ groupName, names: people });
            }
        } catch (e) {
            console.error("Could not save group", e);
        }
    }
    setStep("SCAN");
  };

  // Scanning happens entirely on this device: shrink the photo, read it with the
  // local OCR engine, parse it. Nothing is uploaded, so it also works offline.
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith("image/")) {
        alert("Please upload a valid image file (JPG, PNG).");
        return;
    }

    setLoading(true);
    try {
      const { parsed, prepared } = await scanReceipt(file, setScanStage);
      setPhoto(prepared.display);
      setEnhanced(false);
      setInstructions([]);
      setClarify(null);
      const receipt = receiptToLegacy(parsed.receipt);
      setItems(receipt);
      setScanInfo({
        confidence: parsed.confidence,
        warnings: parsed.warnings,
        itemConf: parsed.items.map((it) => it.confidence),
        empty: parsed.items.length === 0,
        // Only a total that was actually read off a TOTAL line is worth reconciling against.
        printedTotal: parsed.totalSource === "keyword" ? parsed.receipt.total / 100 : null,
      });
      // Even a failed read goes to REVIEW: the user can type the items in, which is
      // better than a dead end.
      setStep("REVIEW");
    } catch (err: any) {
      console.error("Scan Error:", err);
      alert(String(err?.message).startsWith("Couldn't read") ? err.message : "Could not scan that photo. Try again, or add the items by hand.");
      if (fileInputRef.current) fileInputRef.current.value = "";
      if (galleryRef.current) galleryRef.current.value = "";
    } finally {
      setScanStage(null);
      setLoading(false);
    }
  };

  // --- REVIEW edits. The printed total is re-derived from the items + tax, so a
  // corrected price never leaves a stale total (and tax stays whatever the user set).
  const withTotal = (d: ReceiptData): ReceiptData => ({
    ...d,
    total: Math.round((d.items.reduce((sum, it) => sum + it.total_price, 0) + d.tax) * 100) / 100,
  });
  const clearConf = (i: number) => setScanInfo((si) => (si ? { ...si, itemConf: si.itemConf.map((c, k) => (k === i ? 1 : c)) } : si));
  const editItem = (i: number, patch: Partial<ReceiptItem>) => {
    setItems((prev) => {
      if (!prev) return prev;
      const items = prev.items.map((it, k) => {
        if (k !== i) return it;
        const next = { ...it, ...patch };
        next.unit_price = next.quantity > 0 ? Math.round((next.total_price / next.quantity) * 100) / 100 : next.total_price;
        return next;
      });
      return withTotal({ ...prev, items });
    });
    clearConf(i);
  };
  const removeItem = (i: number) => {
    setItems((prev) => (prev ? withTotal({ ...prev, items: prev.items.filter((_, k) => k !== i) }) : prev));
    setScanInfo((si) => (si ? { ...si, itemConf: si.itemConf.filter((_, k) => k !== i) } : si));
  };
  const addItem = () => {
    setItems((prev) =>
      prev ? { ...prev, items: [...prev.items, { name: "", quantity: 1, unit_price: 0, total_price: 0 }] } : prev,
    );
    setScanInfo((si) => (si ? { ...si, itemConf: [...si.itemConf, 1] } : si));
  };
  const editTax = (tax: number) => setItems((prev) => (prev ? withTotal({ ...prev, tax }) : prev));
  const num = (raw: string) => {
    const n = parseFloat(raw.replace(",", "."));
    return Number.isFinite(n) && n >= 0 ? Math.round(n * 100) / 100 : 0;
  };

  // --- Splitting: route -> ambiguity check -> Groq -> Gemini -> on-device rules -> engine.
  const describeTier = (o: Extract<PlanOutcome, { kind: "split" }>) => {
    const why = explainAttempts(o.attempts);
    if (o.tier === "groq") return "Split by Groq using your key.";
    if (o.tier === "gemini") return why ? `${why}, so Gemini split this.` : "Split by Gemini using your key.";
    return why
      ? `${why} — used on-device rules instead.`
      : "Split with on-device rules. Add a free Groq key for smarter splits.";
  };

  const applySplit = async (result: SplitPreview["result"], note: string, next: string[]) => {
    const legacy = splitsToLegacy(result.splits);
    setStructuredSplit(legacy);
    setSplitResult(result.reasoning);
    setInstructions(next);
    setSplitNote(note);
    setClarify(null);
    await attemptSave(legacy, result.reasoning);
    setStep("SUMMARY");
  };

  const runPlan = async (
    next: string[],
    opts: { ignoreAmbiguities?: boolean; peopleOverride?: string[] } = {},
  ) => {
    if (!items) return;
    setLoading(true);
    setClarify(null);
    try {
      const outcome = await planSplit({
        receipt: receiptToDomain(items),
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
      } else if (outcome.kind === "needs-clarification") {
        setClarify({ chips: outcome.chips, preview: outcome.preview, pending: next });
      } else {
        await applySplit(outcome.result, describeTier(outcome), next);
      }
    } catch (err) {
      console.error("Split error:", err);
      alert("Something went wrong splitting this. Try rewording, or check the items.");
    } finally {
      setLoading(false);
    }
  };

  const handleSplit = () => runPlan([instruction]);

  const handleModify = () => {
    const text = modifyText.trim();
    if (!text) return;
    setModifyText("");
    runPlan([...instructions, text]);
  };

  const escapeRe = (t: string) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
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
    // replace: fix the typo / pick the item, then re-run
    const re = new RegExp(`\\b${escapeRe(a.from)}\\b`, "gi");
    const fixed = pending.map((t) => t.replace(re, a.to));
    if (fixed.length === 1) setInstruction(fixed[0]);
    return runPlan(fixed);
  };

  // --- Cloud Enhance: opt-in, per scan, with the user's own Gemini key.
  const handleEnhance = async () => {
    const key = keys.gemini;
    if (!key || !photo) return;
    setEnhancing(true);
    try {
      const parsed = await enhanceReceipt(photo, key, { signal: AbortSignal.timeout(45_000) }) // vision on the free tier measured ~15 s;
      setItems(receiptToLegacy(parsed.receipt));
      setScanInfo({
        confidence: parsed.confidence,
        warnings: parsed.warnings,
        itemConf: parsed.items.map((it) => it.confidence),
        empty: parsed.items.length === 0,
        printedTotal: parsed.totalSource === "keyword" ? parsed.receipt.total / 100 : null,
      });
      setEnhanced(true);
    } catch (err) {
      const kind = err instanceof ProviderError ? err.kind : "network";
      alert(
        kind === "auth" ? "Gemini rejected your key. Check it in AI settings."
        : kind === "rate-limit" ? "Gemini is rate-limited right now. Try again in a minute."
        : kind === "blocked" ? "Gemini declined to read this photo."
        : "Couldn't get a second reading from Gemini. You can still fix the items by hand.",
      );
    } finally {
      setEnhancing(false);
    }
  };

  const attemptSave = async (splitData: any, log: string) => {
     let finalTitle = sessionName.trim();
     if (!finalTitle) {
         // "Session N" needs the server's count; offline we fall back to a date.
         finalTitle = await nextSessionTitle().catch(() => `Session ${new Date().toLocaleDateString()}`);
     }

     // This page still works in RM floats (Phase 6 makes it cents-native), while
     // the database stores integer cents in one standard shape — convert here.
     try {
         const receipt = receiptToDomain(items!);
         await enqueueBill(user?.id ?? null, {
             clientId: sessionClientId,
             billTitle: finalTitle,
             // No merchant name is scanned yet, so this is inferred from the items.
             merchantCategory: inferMerchantCategory(null, receipt.items),
             totalAmount: toCents(displayedTotal),
             currency: symbol,
             data: {
                 split: splitsToDomain(splitData),
                 items: receipt,
                 people,
                 reasoning: log,
             },
         });
     } catch (e) {
         console.error("Save error:", e);
         alert("Failed to save this session on your device. Please try again.");
     }
  };


  // --- FINISH SESSION HANDLER ---
  const handleFinish = () => {
      router.refresh(); 
      router.push("/dashboard");
  };

  return (
    <main className="flex flex-1 flex-col gap-6 p-6 max-w-xl mx-auto w-full mb-20 animate-in fade-in duration-300">
      
      {step !== "NAMES" && (
        <div className="flex items-center justify-between">
          <Button variant="ghost" className="w-fit p-0 h-auto hover:bg-transparent text-slate-500 font-bold uppercase tracking-widest text-[10px]" onClick={() => setStep("NAMES")}>
            <ChevronLeft size={14} className="mr-1" /> Back
          </Button>
          <Button variant="ghost" data-testid="open-ai-settings" className="h-auto p-0 hover:bg-transparent text-slate-500 hover:text-white font-bold uppercase tracking-widest text-[10px]" onClick={() => setKeySheetOpen(true)}>
            <Sparkles size={12} className="mr-1 text-amber-400" /> AI settings{keys.groq || keys.gemini ? " ✓" : ""}
          </Button>
        </div>
      )}
      <Sheet open={keySheetOpen} onOpenChange={setKeySheetOpen}>
        <SheetContent side="bottom" className="max-h-[88vh] overflow-y-auto border-white/10 bg-[#0c0c0e] text-white sm:mx-auto sm:max-w-xl">
          <SheetHeader>
            <SheetTitle className="text-white">AI settings</SheetTitle>
            <SheetDescription className="text-zinc-500">Bring your own free API key for smarter splits.</SheetDescription>
          </SheetHeader>
          <div className="px-4 pb-6">
            <ApiKeySettings />
          </div>
        </SheetContent>
      </Sheet>

      {step === "NAMES" && (
        <div className="space-y-6 animate-in fade-in duration-500">
          <div className="space-y-1">
            <h2 className="text-xl font-bold tracking-tight text-white italic">Group Setup</h2>
            <p className="text-slate-500 text-xs uppercase tracking-widest font-mono">Step 1 of 3</p>
          </div>
          
          <div className="space-y-4">
            <div className="space-y-2">
                <Label className="text-[10px] font-black uppercase text-zinc-500 tracking-widest ml-1">Session Name (Optional)</Label>
                <Input 
                    placeholder="e.g. Friday Dinner" 
                    className="bg-[#0c0c0e] border-white/5 h-12 rounded-xl text-white font-bold"
                    value={sessionName}
                    onChange={(e) => setSessionName(e.target.value)}
                />
            </div>

            {!isGuest && savedGroups.length > 0 && (
                <div className="bg-[#0c0c0e] border border-white/5 rounded-2xl overflow-hidden transition-all">
                    <button 
                        onClick={() => setShowGroupList(!showGroupList)}
                        className="w-full flex items-center justify-between p-3 px-4 text-xs font-bold text-zinc-400 hover:text-white hover:bg-white/5 transition-colors uppercase tracking-widest"
                    >
                        <span className="flex items-center gap-2"><Users size={14}/> Load Saved Group</span>
                        {showGroupList ? <ChevronUp size={14}/> : <ChevronDown size={14}/>}
                    </button>
                    
                    {showGroupList && (
                        <div className="max-h-48 overflow-y-auto border-t border-white/5 bg-black/40">
                            {savedGroups.map((group) => (
                                <button
                                    key={group.id}
                                    onClick={() => loadSavedGroup(group)}
                                    className="w-full text-left p-3 px-4 text-sm text-zinc-300 hover:bg-white/10 hover:text-white border-b border-white/5 last:border-0 flex justify-between items-center group"
                                >
                                    <span className="font-medium">{group.groupName}</span>
                                    <span className="text-[10px] text-zinc-600 font-mono group-hover:text-zinc-400">{group.names.length} people</span>
                                </button>
                            ))}
                        </div>
                    )}
                </div>
            )}
          </div>

          <Card className="bg-[#0c0c0e] border-white/5 shadow-2xl rounded-3xl">
            <CardContent className="pt-6 space-y-4">
              <div className="flex gap-2">
                <Input
                  disabled={!isCreator}
                  className="bg-[#141416] border-white/5 focus:ring-1 ring-white/20 h-11 text-white"
                  placeholder="Enter name..."
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && addPerson()}
                />
                <Button onClick={addPerson} disabled={!isCreator || !newName} className="bg-white text-black hover:bg-zinc-200">
                  <Plus size={18} />
                </Button>
              </div>
              <div className="flex flex-wrap gap-2 min-h-[44px] p-2 rounded-lg bg-black/20 border border-white/5">
                {people.map((p) => (
                  <Badge key={p} variant="secondary" className="bg-white/5 py-1.5 px-3 flex gap-2 items-center border-none text-slate-300">
                    {p}
                    {isCreator && (
                      <button onClick={() => removePerson(p)} className="hover:bg-white/20 rounded-full p-0.5 transition-colors"><X size={12} /></button>
                    )}
                  </Badge>
                ))}
              </div>

              {!isGuest && (
                <div className="space-y-3 pt-2">
                  {activeGroupId && hasGroupChanged() ? (
                      <div className="bg-amber-500/10 border border-amber-500/20 rounded-xl p-3 animate-in fade-in">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <RefreshCw size={14} className="text-amber-500"/>
                                <span className="text-xs text-amber-500 font-bold uppercase tracking-tight">Update saved group?</span>
                            </div>
                            <Switch checked={updateGroup} onCheckedChange={setUpdateGroup} className="data-[state=checked]:bg-amber-500"/>
                          </div>
                          <p className="text-[10px] text-zinc-500 mt-1 pl-6">
                              Update <strong>{groupName}</strong> with these changes?
                          </p>
                      </div>
                  ) : !activeGroupId ? (
                      <>
                        <div className="flex items-center justify-between px-1">
                            <span className="text-xs text-slate-400 font-medium tracking-tight">Save this group?</span>
                            <Switch checked={saveThisGroup} onCheckedChange={setSaveThisGroup} />
                        </div>
                        {saveThisGroup && (
                            <Input 
                            placeholder="Group Name (e.g. Weekend Crew)" 
                            value={groupName}
                            onChange={(e) => setGroupName(e.target.value)}
                            className="bg-[#141416] border-white/5 h-10 text-xs text-white"
                            />
                        )}
                      </>
                  ) : (
                       <div className="flex items-center justify-center gap-2 p-2 opacity-50">
                            <Users size={12} className="text-zinc-500"/>
                            <span className="text-[10px] text-zinc-500 uppercase tracking-widest font-bold">Loaded: {groupName}</span>
                       </div>
                  )}
                </div>
              )}

              <Button className="w-full h-12 bg-white text-black font-black uppercase tracking-tighter rounded-xl" disabled={people.length < 1} onClick={handleStartScanning}>
                Start Scanning
              </Button>
            </CardContent>
          </Card>
        </div>
      )}

      {step === "SCAN" && (
        <div className="space-y-6 animate-in slide-in-from-bottom-4">
          <Card className="bg-[#0c0c0e] border-dashed border border-white/10 py-12 rounded-3xl">
            <CardContent className="text-center space-y-6">
              <div className="w-16 h-16 bg-white/5 rounded-full flex items-center justify-center mx-auto border border-white/10">
                <Receipt className="w-6 h-6 text-white opacity-40" />
              </div>
              <h3 className="text-lg font-bold uppercase tracking-tighter text-white">Scan Receipt</h3>
              
              <input 
                  type="file" 
                  accept="image/*" 
                  capture="environment" 
                  ref={fileInputRef} 
                  className="hidden" 
                  onChange={handleFileUpload}
                  onClick={(e: any) => e.target.value = null} 
              />
              <input 
                  type="file" 
                  accept="image/*" 
                  ref={galleryRef} 
                  className="hidden" 
                  onChange={handleFileUpload}
                  onClick={(e: any) => e.target.value = null}
              />
              
              <div className="flex flex-col gap-2 px-4">
                <Button className="w-full h-14 text-md bg-white text-black font-bold rounded-xl" onClick={() => fileInputRef.current?.click()} disabled={loading || !isCreator}>
                  {loading ? (<><Loader2 className="animate-spin mr-2" />{scanStage === "preparing" ? "Preparing photo…" : scanStage === "loading-reader" ? "Loading reader (first time only)…" : "Reading receipt…"}</>) : "Snap Photo"}
                </Button>
                <Button variant="ghost" className="text-xs text-slate-500 hover:text-white uppercase font-bold tracking-widest" onClick={() => galleryRef.current?.click()} disabled={loading || !isCreator}>
                  <ImageIcon size={14} className="mr-2" /> Gallery
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {step === "REVIEW" && (
        <div className="space-y-6">
          {scanInfo && (scanInfo.empty || scanInfo.confidence < 0.6 || scanInfo.warnings.some((w) => !w.startsWith("Items add up"))) && (
            <div data-testid="scan-warning" className="flex gap-3 rounded-2xl border border-amber-500/30 bg-amber-500/5 p-4 text-amber-200">
              <TriangleAlert className="mt-0.5 h-4 w-4 shrink-0" />
              <div className="space-y-1 text-xs">
                <p className="font-bold">
                  {scanInfo.empty
                    ? "We couldn't read any items from that photo."
                    : "Please check these numbers against your receipt."}
                </p>
                {scanInfo.empty && <p className="opacity-80">Add the items below, or go back and try a clearer photo.</p>}
                {scanInfo.warnings.filter((w) => !w.startsWith("Items add up")).slice(0, 3).map((w, i) => (
                  <p key={i} className="opacity-80">{w}</p>
                ))}
              </div>
            </div>
          )}
          {(() => {
            const diff = scanInfo?.printedTotal != null && items ? Math.round((scanInfo.printedTotal - (subtotal + items.tax)) * 100) / 100 : 0;
            const unsure = !!scanInfo && (scanInfo.empty || scanInfo.confidence < 0.6 || Math.abs(diff) >= 0.01);
            if (!unsure || !photo || enhanced) return null;
            return keys.gemini ? (
              <div data-testid="enhance-card" className="space-y-2 rounded-2xl border border-white/10 bg-white/[0.03] p-4">
                <p className="text-xs font-bold text-white">Not sure about this scan?</p>
                <p className="text-xs text-zinc-500">Ask Gemini to re-read the photo. This sends the photo to Google using your own key.</p>
                <Button type="button" variant="ghost" onClick={handleEnhance} disabled={enhancing} className="h-9 rounded-full border border-white/10 bg-white/5 px-4 text-[11px] font-bold text-white hover:bg-white/10 hover:text-white">
                  {enhancing ? <><Loader2 className="mr-2 h-3 w-3 animate-spin" /> Reading…</> : "Re-read with Gemini"}
                </Button>
              </div>
            ) : (
              <p data-testid="enhance-hint" className="px-1 text-[11px] text-zinc-500">
                Scan looks shaky.{" "}
                <button type="button" onClick={() => setKeySheetOpen(true)} className="font-bold text-indigo-400 hover:text-indigo-300">Add a free Gemini key</button>{" "}
                to get a second opinion, or fix the numbers by hand.
              </p>
            );
          })()}
          <Card className="bg-[#0c0c0e] border-white/5 overflow-hidden shadow-2xl rounded-3xl">
            <div className="bg-white/5 p-4 border-b border-white/5 text-[10px] font-bold uppercase text-slate-500 tracking-widest">Extracted Items · tap to edit</div>
            <CardContent className="p-0">
              <div className="divide-y divide-white/5">
                {items?.items.map((item, i) => {
                  const unsure = (scanInfo?.itemConf[i] ?? 1) < 0.6;
                  return (
                    <div
                      key={i}
                      data-testid="review-row"
                      data-unsure={unsure ? "true" : undefined}
                      className={`flex items-center gap-2 px-4 py-3 ${unsure ? "bg-amber-500/5 ring-1 ring-inset ring-amber-500/30" : ""}`}
                    >
                      <Input
                        aria-label="Item name"
                        value={item.name}
                        placeholder="Item"
                        onChange={(e) => editItem(i, { name: e.target.value })}
                        disabled={!isCreator}
                        className="h-9 flex-1 border-white/5 bg-black text-sm text-zinc-200"
                      />
                      <Input
                        aria-label="Quantity"
                        key={`q${i}-${item.quantity}`}
                        type="number"
                        min={1}
                        step={1}
                        defaultValue={item.quantity}
                        onBlur={(e) => editItem(i, { quantity: Math.max(1, Math.round(num(e.target.value))) })}
                        disabled={!isCreator}
                        className="h-9 w-14 border-white/5 bg-black text-center text-sm text-slate-400"
                      />
                      <Input
                        aria-label="Price"
                        key={`p${i}-${item.total_price}`}
                        type="number"
                        min={0}
                        step={0.01}
                        defaultValue={item.total_price.toFixed(2)}
                        onBlur={(e) => editItem(i, { total_price: num(e.target.value) })}
                        disabled={!isCreator}
                        className="h-9 w-24 border-white/5 bg-black text-right font-mono text-sm text-white"
                      />
                      <button
                        type="button"
                        aria-label="Remove item"
                        onClick={() => removeItem(i)}
                        className="p-1 text-zinc-600 transition-colors hover:text-red-500"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  );
                })}
              </div>
              {scanInfo?.printedTotal != null && items && items.items.length > 0 && (() => {
                const mine = Math.round((subtotal + items.tax) * 100) / 100;
                const diff = Math.round((scanInfo.printedTotal - mine) * 100) / 100;
                return Math.abs(diff) < 0.005 ? (
                  <p data-testid="reconcile" data-ok="true" className="px-4 pt-3 text-[11px] font-bold text-emerald-400">
                    ✓ Items + tax match the receipt total ({symbol}{scanInfo.printedTotal.toFixed(2)})
                  </p>
                ) : (
                  <p data-testid="reconcile" className="px-4 pt-3 text-[11px] font-bold text-amber-300">
                    Receipt says {symbol}{scanInfo.printedTotal.toFixed(2)}; items + tax come to {symbol}{mine.toFixed(2)} ({diff > 0 ? `${symbol}${diff.toFixed(2)} missing` : `${symbol}${(-diff).toFixed(2)} too much`}).
                  </p>
                );
              })()}
              <div className="px-4 pt-3">
                <Button type="button" variant="ghost" onClick={addItem} className="h-9 text-[10px] font-bold uppercase tracking-widest text-slate-500 hover:text-white">
                  <Plus className="mr-1 h-3 w-3" /> Add item
                </Button>
              </div>
              <div className="p-6 space-y-4">
                <div className="flex items-center justify-between p-4 rounded-xl bg-black border border-white/5">
                  <div className="space-y-0.5">
                    <div className="text-xs text-white opacity-60 font-mono tracking-tighter uppercase">Apply Tax & Svc</div>
                    <div className="flex items-center gap-1 text-[10px] text-zinc-600 font-bold uppercase italic">
                      +{symbol}
                      <Input
                        aria-label="Tax and service"
                        key={`tax-${items?.tax}`}
                        type="number"
                        min={0}
                        step={0.01}
                        defaultValue={(items?.tax ?? 0).toFixed(2)}
                        onBlur={(e) => editTax(num(e.target.value))}
                        disabled={!isCreator}
                        className="h-7 w-20 border-white/5 bg-black px-2 font-mono text-[11px] text-zinc-400"
                      />
                    </div>
                  </div>
                  <Switch checked={includeTax} onCheckedChange={setIncludeTax} disabled={!isCreator} />
                </div>

                <div className="flex justify-between items-center px-2 py-2 border-t border-white/5 pt-4">
                  <span className="text-[10px] font-black uppercase tracking-[0.2em] text-zinc-500">Total to Split</span>
                  <span data-testid="review-total" className="text-xl font-black font-mono italic text-white tracking-tighter">{symbol}{displayedTotal.toFixed(2)}</span>
                </div>

                <Input disabled={!isCreator} placeholder="Instructions (e.g. Split equally)" value={instruction} onChange={(e) => setInstruction(e.target.value)} onKeyDown={(e) => e.key === "Enter" && !loading && handleSplit()} className="bg-black border-white/5 h-12 text-white" />
                {clarify && <ClarifyChips chips={clarify.chips} hasPreview={!!clarify.preview} onAction={handleClarify} />}
                {splitNote && !clarify && <p data-testid="split-note" className="px-1 text-[11px] text-zinc-500">{splitNote}</p>}
                <Button className="w-full h-12 bg-white text-black font-black uppercase tracking-tight rounded-xl" onClick={handleSplit} disabled={loading || !isCreator || !items || items.items.length === 0 || displayedTotal <= 0}>
                  {loading ? <Loader2 className="animate-spin" /> : "Split Bill"}
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {step === "SUMMARY" && (
        <div className="space-y-6 animate-in zoom-in-95 duration-300">
          <Card className="bg-[#0c0c0e] border-white/5 shadow-2xl overflow-hidden ring-1 ring-white/10 rounded-3xl">
            <CardHeader className="text-center border-b border-white/5 py-4 bg-white/[0.02]">
              <CardTitle className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.3em]">Final Settlement</CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <Table>
                <TableBody>
                  {structuredSplit.map((row, i) => (
                    <TableRow key={i} className="border-white/5">
                      <TableCell className="py-5 font-bold text-sm text-white px-8">{row.name}</TableCell>
                      <TableCell className="text-right font-mono text-white text-xl px-8">{symbol}{row.amount.toFixed(2)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
              <div className="border-t border-white/5">
                <button onClick={() => setShowReasoning(!showReasoning)} className="w-full p-4 flex justify-between items-center text-[10px] font-bold text-zinc-500 hover:bg-white/5 transition-colors uppercase tracking-widest">
                  <div className="flex items-center gap-2"><ScrollText size={14} /> System Reasoning Log</div>
                  {showReasoning ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                </button>
                {showReasoning && (
                  <div className="px-6 pb-6 animate-in slide-in-from-top-2 duration-200 space-y-4">
                    <div className="p-4 bg-black rounded-xl text-[10px] font-mono text-zinc-500 whitespace-pre-wrap leading-relaxed border border-white/5 max-h-60 overflow-y-auto italic text-left">{splitResult}</div>
                  </div>
                )}
              </div>
              <div className="space-y-3 border-t border-white/5 p-6">
                <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-widest text-zinc-500">
                  <Sparkles size={12} className="text-amber-400" /> Change something
                </div>
                <div className="flex gap-2">
                  <Input
                    data-testid="modify-input"
                    value={modifyText}
                    onChange={(e) => setModifyText(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && !loading && handleModify()}
                    placeholder="e.g. Aisha didn't have the rice"
                    disabled={loading}
                    className="h-11 border-white/5 bg-black text-white"
                  />
                  <Button data-testid="modify-apply" type="button" onClick={handleModify} disabled={loading || !modifyText.trim()} className="h-11 bg-white px-4 text-[10px] font-black uppercase tracking-widest text-black hover:bg-zinc-200">
                    {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : "Apply"}
                  </Button>
                </div>
                {clarify && <ClarifyChips chips={clarify.chips} hasPreview={!!clarify.preview} onAction={handleClarify} />}
                {splitNote && !clarify && <p data-testid="split-note" className="text-[11px] text-zinc-500">{splitNote}</p>}
              </div>
              <div className="p-6 pt-0 space-y-3">
                <Button className="w-full h-12 bg-[#25D366] text-black font-black rounded-xl uppercase tracking-tighter" onClick={() => {
                  let text = `*Bill-a Summary (${symbol})*\n\n`;
                  structuredSplit.forEach((r) => (text += `👤 *${r.name}*: ${symbol}${r.amount.toFixed(2)}\n`));
                  window.open(`whatsapp://send?text=${encodeURIComponent(text)}`);
                }}>
                  <Share2 className="w-4 h-4 mr-2" /> Share via WhatsApp
                </Button>
                <Button variant="outline" className="w-full h-12 border-white/5 text-zinc-500 font-bold rounded-xl uppercase tracking-widest text-[10px]" onClick={handleFinish}>
                  Finish Session
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </main>
  );
}

export default function BillSplitter() {
  return (
    <Suspense fallback={<div className="flex min-h-screen items-center justify-center bg-black"><Loader2 className="animate-spin w-8 h-8 text-white opacity-20" /></div>}>
      <BillSplitterContent />
    </Suspense>
  );
}