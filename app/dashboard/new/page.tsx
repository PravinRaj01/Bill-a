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
import { createClient } from "@/utils/supabase/client";
import { useRouter, useSearchParams } from "next/navigation";
import {
  Loader2,
  Plus,
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

const API_URL = "https://favourable-eunice-pravinraj-code-24722b81.koyeb.app";

function BillSplitterContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const supabase = createClient();

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

  // GROUP SAVING STATE
  const [saveThisGroup, setSaveThisGroup] = useState(false);
  const [updateGroup, setUpdateGroup] = useState(true); 
  const [groupName, setGroupName] = useState("");
  const [activeGroupId, setActiveGroupId] = useState<string | null>(null);
  const [originalPeople, setOriginalPeople] = useState<string[]>([]); 
  
  // GROUP LOADING UI STATE
  const [savedGroups, setSavedGroups] = useState<any[]>([]);
  const [showGroupList, setShowGroupList] = useState(false);

  const isCreator = true;
  const fileInputRef = useRef<HTMLInputElement>(null);
  const galleryRef = useRef<HTMLInputElement>(null);

  const subtotal = items?.items.reduce((sum, item) => sum + item.total_price, 0) || 0;
  const displayedTotal = includeTax ? items?.total || 0 : subtotal;

  // 1. Check User & Fetch All Saved Groups
  useEffect(() => {
    const init = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      if (session) { 
        setUser(session.user); 
        setIsGuest(false);
        const { data: groups } = await supabase
            .from('saved_groups')
            .select('*')
            .eq('user_id', session.user.id)
            .order('created_at', { ascending: false });
        if (groups) setSavedGroups(groups);
      } else { 
        setIsGuest(true); 
      }
    };
    init();
  }, [supabase]);

  // 2. MAIN RESTORE LOGIC
  useEffect(() => {
    const restoreFromChat = searchParams.get("restore_from_chat");
    const restoreFromHistory = searchParams.get("restore_from_history");
    const namesParam = searchParams.get("names");
    const groupId = searchParams.get("group_id");

    // A. RESTORE FROM CHAT (User clicked "Save" in chat)
    if (restoreFromChat) {
        const chatResult = sessionStorage.getItem("billa_chat_result");
        const context = sessionStorage.getItem("billa_chat_context"); 
        
        if (chatResult && context) {
            const { splits, reasoning } = JSON.parse(chatResult);
            const { items: origItems, people_list } = JSON.parse(context);
            
            setItems(origItems);
            setPeople(people_list);
            setStructuredSplit(splits);
            setSplitResult(reasoning);
            setStep("SUMMARY"); 
            
            sessionStorage.removeItem("billa_chat_result");
            // Also update the snapshot so refreshing works
            sessionStorage.setItem("billa_snapshot", JSON.stringify({
                 items: origItems, people: people_list, structuredSplit: splits, splitResult: reasoning
            }));
        }
        return;
    }

    // B. RESTORE FROM HISTORY (User clicked "Continue" in History)
    if (restoreFromHistory) {
        const historyData = sessionStorage.getItem("billa_restore_data");
        if (historyData) {
            const { data, currency } = JSON.parse(historyData);
            
            if (data.items && data.split) {
                setItems(data.items);
                setPeople(data.people || []);
                setStructuredSplit(data.split);
            } else {
                const legacySplit = Array.isArray(data) ? data : data.splits || [];
                setStructuredSplit(legacySplit);
                setPeople(legacySplit.map((p:any) => p.name));
                setItems({ items: [], total: 0, tax: 0, currency: currency }); 
            }
            
            setSplitResult(data.reasoning || "Restored from history.");
            setStep("SUMMARY");
            sessionStorage.removeItem("billa_restore_data");
            // Clear snapshot so we don't mix old drafts
            sessionStorage.removeItem("billa_snapshot");
        }
        return;
    }

    // C. NEW SESSION (URL Params)
    if (namesParam || groupId) {
        // Clear old snapshots because we are explicitly starting something new
        sessionStorage.removeItem("billa_snapshot");
        
        if (namesParam) {
            const loadedNames = decodeURIComponent(namesParam).split(",");
            setPeople(loadedNames);
        } else if (groupId) {
            const loadGroup = async () => {
                const { data } = await supabase.from("saved_groups").select("names, group_name, id").eq("id", groupId).single();
                if (data) {
                setPeople(data.names);
                setGroupName(data.group_name);
                setActiveGroupId(data.id);
                setOriginalPeople(data.names);
                }
            };
            loadGroup();
        }
        return;
    }

    // D. CANCEL/BACK BUTTON FALLBACK (Snapshot Restore)
    // If we have no specific instructions, check if we have an active session snapshot
    const snapshot = sessionStorage.getItem("billa_snapshot");
    if (snapshot) {
        try {
            const data = JSON.parse(snapshot);
            // Only restore if it looks valid
            if (data.items && data.structuredSplit) {
                setItems(data.items);
                setPeople(data.people);
                setStructuredSplit(data.structuredSplit);
                setSplitResult(data.splitResult);
                setStep("SUMMARY");
            }
        } catch (e) {
            console.error("Failed to restore snapshot", e);
        }
    }

  }, [searchParams, supabase]);

  const hasGroupChanged = () => {
      if (!activeGroupId) return false;
      if (people.length !== originalPeople.length) return true;
      const sortedPeople = [...people].sort();
      const sortedOriginal = [...originalPeople].sort();
      return JSON.stringify(sortedPeople) !== JSON.stringify(sortedOriginal);
  };

  const loadSavedGroup = (group: any) => {
      setPeople(group.names);
      setGroupName(group.group_name);
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
        if (activeGroupId && hasGroupChanged() && updateGroup) {
             await supabase.from("saved_groups")
                .update({ names: people })
                .eq("id", activeGroupId);
        } else if (!activeGroupId && saveThisGroup && groupName) {
            await supabase.from("saved_groups").insert({
                user_id: user.id,
                group_name: groupName,
                names: people,
            });
        }
    }
    setStep("SCAN");
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith("image/")) {
        alert("Please upload a valid image file (JPG, PNG).");
        return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 45000); 

    try {
      const res = await fetch(`${API_URL}/scan`, { 
        method: "POST", 
        body: formData,
        signal: controller.signal
      });
      clearTimeout(timeoutId);

      if (!res.ok) {
          throw new Error(`Server responded with ${res.status}`);
      }

      const data = await res.json();
      
      if (!data.items || !Array.isArray(data.items) || data.items.length === 0) {
        alert("Could not detect any receipt items. \n\nPlease ensure:\n1. The photo is clear and well-lit\n2. It is a valid receipt (not a random selfie!)\n3. Prices and item names are visible.");
        setLoading(false);
        if (fileInputRef.current) fileInputRef.current.value = "";
        if (galleryRef.current) galleryRef.current.value = "";
        return;
      }

      setItems(data);
      setStep("REVIEW");

    } catch (err: any) {
      console.error("Scan Error:", err);
      if (err.name === 'AbortError') {
        alert("Scanning timed out. The server might be waking up (cold start). Please try again in a few seconds.");
      } else {
        alert("Failed to scan receipt. Please check your connection and try again.");
      }
      
      if (fileInputRef.current) fileInputRef.current.value = "";
      if (galleryRef.current) galleryRef.current.value = "";
    } finally {
        setLoading(false);
    }
  };

  const handleSplit = async () => {
    setLoading(true);
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 45000);

    try {
      const res = await fetch(`${API_URL}/split`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          receipt_data: JSON.stringify(items),
          user_instruction: instruction || "Split equally",
          people_list: people,
          apply_tax: includeTax,
        }),
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);

      if (!res.ok) throw new Error(`Server Error: ${res.status}`);

      const data = await res.json();
      
      try {
        const cleanJson = data.result.replace(/```json/g, "").replace(/```/g, "").trim();
        const parsedResult = JSON.parse(cleanJson);

        if (parsedResult.splits && Array.isArray(parsedResult.splits)) {
            setStructuredSplit(parsedResult.splits);
            setSplitResult(parsedResult.reasoning || "Calculation complete.");
            await attemptSave(parsedResult.splits, parsedResult.reasoning || data.result);
            setStep("SUMMARY");
        } 
        else if (Array.isArray(parsedResult)) {
            setStructuredSplit(parsedResult);
            setSplitResult("Split successful.");
            await attemptSave(parsedResult, data.result);
            setStep("SUMMARY");
        } 
        else {
            throw new Error("AI response was not a valid split list.");
        }

      } catch (parseError) {
          console.error("JSON Parse Error:", parseError);
          throw new Error("Failed to read AI response. Please try splitting again.");
      }

    } catch (err: any) {
      console.error("Split Error:", err);
      if (err.name === 'AbortError') {
        alert("Split timed out. The server is waking up, please try clicking Split again.");
      } else {
        alert("Split failed. Please ensure your instructions are clear or try again.");
      }
    } finally {
        setLoading(false);
    }
  };

  const attemptSave = async (splitData: any, log: string) => {
    if (user) {
        await saveToHistory(splitData, log);
    }
  };

  const saveToHistory = async (splitData: any, log: string) => {
     let finalTitle = sessionName.trim();
     if (!finalTitle) {
         const { data: userBills } = await supabase.from("bill_history").select("id").eq("user_id", user.id);
         const sessionNum = (userBills?.length || 0) + 1;
         finalTitle = `Session ${sessionNum}`;
     }

     const richData = {
         split: splitData,
         items: items,    
         people: people,  
         reasoning: log
     };

     const { error } = await supabase.from("bill_history").insert({
         user_id: user.id,
         bill_title: finalTitle,
         total_amount: displayedTotal,
         currency: symbol, 
         data: richData, 
         reasoning_log: log,
     });

     if (error) {
         console.error("Supabase Save Error:", error);
         alert(`Failed to save history: ${error.message}`);
     }
  };

  const handleModifyInChat = () => {
    // 1. Save context for AI Logic
    const contextData = JSON.stringify({
        items: items, 
        people_list: people,
        current_instruction: instruction
    });
    sessionStorage.setItem("billa_chat_context", contextData);

    // 2. NEW: Save Visual Snapshot for the "Cancel" / Back button
    // This allows Step D in useEffect to restore the view when coming back
    sessionStorage.setItem("billa_snapshot", JSON.stringify({
        items,
        people,
        structuredSplit,
        splitResult,
        step: "SUMMARY"
    }));

    // 3. Prompt setup
    const initialPrompt = instruction 
        ? `I tried to split the bill with instruction: "${instruction}", but I need to make changes.` 
        : `I split the bill equally, but I need to make specific adjustments.`;
    
    sessionStorage.setItem("billa_initial_prompt", initialPrompt);
    
    router.push(`/dashboard/chat`);
  };

  // --- FINISH SESSION HANDLER ---
  const handleFinish = () => {
      // Clear all snapshots to prevent stale data loading next time
      sessionStorage.removeItem("billa_snapshot");
      sessionStorage.removeItem("billa_chat_result");
      sessionStorage.removeItem("billa_restore_data");
      
      router.refresh(); 
      router.push("/dashboard");
  };

  return (
    <main className="flex flex-1 flex-col gap-6 p-6 max-w-xl mx-auto w-full mb-20 animate-in fade-in duration-300">
      
      {step !== "NAMES" && (
        <Button variant="ghost" className="w-fit p-0 h-auto hover:bg-transparent text-slate-500 font-bold uppercase tracking-widest text-[10px]" onClick={() => setStep("NAMES")}>
          <ChevronLeft size={14} className="mr-1" /> Back
        </Button>
      )}

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
                                    <span className="font-medium">{group.group_name}</span>
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
                  {loading ? <Loader2 className="animate-spin mr-2" /> : "Snap Photo"}
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
          <Card className="bg-[#0c0c0e] border-white/5 overflow-hidden shadow-2xl rounded-3xl">
            <div className="bg-white/5 p-4 border-b border-white/5 text-[10px] font-bold uppercase text-slate-500 tracking-widest">Extracted Items</div>
            <CardContent className="p-0">
              <Table>
                <TableBody>
                  {items?.items.map((item, i) => (
                    <TableRow key={i} className="border-white/5">
                      <TableCell className="py-4 text-sm font-medium text-zinc-200">{item.name}</TableCell>
                      <TableCell className="text-center text-sm text-slate-500">x{item.quantity}</TableCell>
                      <TableCell className="text-right font-mono text-white">{item.total_price.toFixed(2)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
              <div className="p-6 space-y-4">
                <div className="flex items-center justify-between p-4 rounded-xl bg-black border border-white/5">
                  <div className="space-y-0.5">
                    <div className="text-xs text-white opacity-60 font-mono tracking-tighter uppercase">Apply Tax & Svc</div>
                    <div className="text-[10px] text-zinc-600 font-bold uppercase italic">+{symbol}{items?.tax.toFixed(2)}</div>
                  </div>
                  <Switch checked={includeTax} onCheckedChange={setIncludeTax} disabled={!isCreator} />
                </div>

                <div className="flex justify-between items-center px-2 py-2 border-t border-white/5 pt-4">
                  <span className="text-[10px] font-black uppercase tracking-[0.2em] text-zinc-500">Total to Split</span>
                  <span className="text-xl font-black font-mono italic text-white tracking-tighter">{symbol}{displayedTotal.toFixed(2)}</span>
                </div>

                <Input disabled={!isCreator} placeholder="Instructions (e.g. Split equally)" value={instruction} onChange={(e) => setInstruction(e.target.value)} className="bg-black border-white/5 h-12 text-white" />
                <Button className="w-full h-12 bg-white text-black font-black uppercase tracking-tight rounded-xl" onClick={handleSplit} disabled={loading || !isCreator}>
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
                    
                    <Button 
                        variant="ghost" 
                        onClick={handleModifyInChat}
                        className="w-full h-10 border border-white/10 bg-white/5 text-xs text-white hover:bg-white/10 hover:text-white uppercase tracking-wider font-bold rounded-xl"
                    >
                        <MessageSquare size={14} className="mr-2 text-indigo-400"/>
                        Modify with AI Chat
                        <Sparkles size={12} className="ml-2 text-amber-400"/>
                    </Button>
                  </div>
                )}
              </div>
              <div className="p-6 space-y-3">
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