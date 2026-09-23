"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Send, Bot, User, Loader2, Save, X } from "lucide-react"; // Imported X for Cancel

const API_URL = "https://favourable-eunice-pravinraj-code-24722b81.koyeb.app";

interface Message {
  role: "user" | "assistant";
  content: string;
  splitData?: any[]; 
}

export default function ChatPage() {
  const router = useRouter();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [receiptContext, setReceiptContext] = useState<string>("");
  const [latestSplit, setLatestSplit] = useState<any[] | null>(null); // Track latest valid split
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const data = sessionStorage.getItem("billa_chat_context");
    const initialPrompt = sessionStorage.getItem("billa_initial_prompt");
    
    if (data) {
      setReceiptContext(data);
      setMessages([
        { role: "assistant", content: "I have the current bill details. How would you like to modify the split?" }
      ]);
      
      if (initialPrompt) {
         handleSend(initialPrompt);
         sessionStorage.removeItem("billa_initial_prompt"); 
      }
    } else {
      router.push("/dashboard"); 
    }
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  const handleSend = async (text: string) => {
    if (!text.trim()) return;

    const userMsg: Message = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${API_URL}/chat_modify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          receipt_data: receiptContext,
          history: messages.map(m => `${m.role}: ${m.content}`),
          user_message: text
        }),
      });

      if (!res.ok) throw new Error("Failed to chat");

      const data = await res.json();
      
      const aiMsg: Message = { 
          role: "assistant", 
          content: data.reply,
          splitData: data.splits 
      };

      if (data.splits) {
          setLatestSplit(data.splits); // Capture the new split
      }

      setMessages((prev) => [...prev, aiMsg]);

    } catch (err) {
      console.error(err);
      setMessages((prev) => [...prev, { role: "assistant", content: "Sorry, I had trouble connecting. Please try again." }]);
    } finally {
      setLoading(false);
    }
  };

  // 1. CANCEL = Discard Changes
  const handleCancel = () => {
      router.back(); 
  };

  // 2. SAVE = Commit Changes
  const handleSave = () => {
      if (latestSplit) {
          sessionStorage.setItem("billa_chat_result", JSON.stringify({
              splits: latestSplit,
              reasoning: messages[messages.length - 1].content 
          }));
          router.push("/dashboard/new?restore_from_chat=true");
      }
  };

  return (
    <div className="flex flex-col h-screen bg-black text-white max-w-xl mx-auto border-x border-white/5">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-white/10 bg-[#0c0c0e]">
        
        {/* LEFT: Cancel Button */}
        <Button 
            variant="ghost" 
            onClick={handleCancel} 
            className="h-8 text-zinc-400 hover:text-white hover:bg-white/10 text-xs font-bold uppercase tracking-wider px-2"
        >
            <X size={16} className="mr-1.5" /> Cancel
        </Button>
        
        {/* CENTER: Status */}
        <div className="text-center">
            <h1 className="font-bold text-sm">Bill-a Assistant</h1>
            <p className="text-[10px] text-zinc-500 flex items-center justify-center gap-1">
               <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"/> Online
            </p>
        </div>
        
        {/* RIGHT: Save Button (Only shows if there is a NEW split to save) */}
        <div className="w-[84px] flex justify-end"> 
            {latestSplit ? (
                <Button 
                    onClick={handleSave}
                    className="h-8 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold uppercase tracking-wider rounded-lg shadow-lg shadow-indigo-900/20"
                >
                    <Save size={14} className="mr-1.5" /> Save
                </Button>
            ) : (
                <div className="w-8" /> // Spacer to keep title centered
            )}
        </div>
      </div>

      {/* Chat Area */}
      <ScrollArea className="flex-1 p-4 space-y-4">
        <div className="flex flex-col gap-4 pb-4">
          {messages.map((m, i) => (
            <div key={i} className={`flex gap-3 ${m.role === "user" ? "justify-end" : "justify-start"}`}>
              {m.role === "assistant" && (
                <div className="w-6 h-6 rounded-full bg-zinc-800 flex items-center justify-center mt-1 shrink-0">
                  <Bot size={12} className="text-zinc-400" />
                </div>
              )}
              
              <div className={`max-w-[85%] space-y-2 ${m.role === "user" ? "items-end flex flex-col" : ""}`}>
                <div className={`p-3 rounded-2xl text-sm leading-relaxed ${
                  m.role === "user" 
                    ? "bg-indigo-600 text-white rounded-tr-sm" 
                    : "bg-[#1c1c1e] text-zinc-200 border border-white/5 rounded-tl-sm"
                }`}>
                  {m.content}
                </div>

                {m.splitData && (
                    <Card className="w-full bg-black border border-white/10 overflow-hidden animate-in zoom-in-95 mt-2 shadow-2xl">
                        <div className="bg-white/5 px-3 py-2 border-b border-white/5 flex justify-between items-center">
                            <span className="text-[10px] font-bold uppercase tracking-widest text-indigo-400">Updated Bill</span>
                        </div>
                        <CardContent className="p-0">
                            {m.splitData.map((row: any, idx: number) => (
                                <div key={idx} className="flex justify-between p-3 border-b border-white/5 text-xs last:border-0 hover:bg-white/5 transition-colors">
                                    <span className="font-medium text-white">{row.name}</span>
                                    <span className="font-mono text-zinc-400">{row.amount.toFixed(2)}</span>
                                </div>
                            ))}
                        </CardContent>
                    </Card>
                )}
              </div>
            </div>
          ))}
          
          {loading && (
             <div className="flex gap-3">
                <div className="w-6 h-6 rounded-full bg-zinc-800 flex items-center justify-center mt-1">
                  <Bot size={12} className="text-zinc-400" />
                </div>
                <div className="bg-[#1c1c1e] p-3 rounded-2xl rounded-tl-sm border border-white/5 flex items-center gap-2">
                    <Loader2 className="w-3 h-3 animate-spin text-zinc-500" />
                    <span className="text-xs text-zinc-500 animate-pulse">Thinking...</span>
                </div>
             </div>
          )}
          <div ref={scrollRef} />
        </div>
      </ScrollArea>

      {/* Input Area */}
      <div className="p-4 bg-[#0c0c0e] border-t border-white/10">
        <form onSubmit={(e) => { e.preventDefault(); handleSend(input); }} className="flex gap-2">
          <Input 
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your modification..."
            className="bg-[#1c1c1e] border-white/5 focus:ring-indigo-500/50 text-white"
          />
          <Button type="submit" size="icon" className="bg-white text-black hover:bg-zinc-200" disabled={loading || !input.trim()}>
            <Send size={18} />
          </Button>
        </form>
      </div>
    </div>
  );
}