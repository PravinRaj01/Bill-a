"use client";

import { useEffect, useState } from "react";
import { Loader2, ShieldCheck } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { getKeys, looksLikeKey, maskKey, onKeysChanged, removeKey, setKey, type Keys } from "@/lib/ai/keyStore";
import { testKey, type KeyStatus } from "@/lib/ai/providers/testKey";
import type { ProviderId } from "@/lib/ai/providers/types";

// Where the user pastes their own Groq / Gemini key. Keys stay in this browser's
// localStorage (lib/ai/keyStore.ts) and are sent only to the provider — never to
// Bill.a's servers. Both are optional: without a key the app splits with its
// on-device rules.

const PROVIDERS: {
  id: ProviderId;
  name: string;
  blurb: string;
  href: string;
  placeholder: string;
  note?: string;
}[] = [
  {
    id: "groq",
    name: "Groq",
    blurb: "Smarter splits, fast. Free tier.",
    href: "https://console.groq.com/keys",
    placeholder: "gsk_…",
  },
  {
    id: "gemini",
    name: "Google Gemini",
    blurb: "Backup for splits, and can re-read a blurry receipt photo.",
    href: "https://aistudio.google.com/apikey",
    placeholder: "AIza…",
    note:
      "On Google's free tier, prompts (your receipt text, and photos if you use “Re-read with Gemini”) may be used to improve Google's products and reviewed by people. " +
      "The free tier isn't available in the EEA, UK or Switzerland, and you must be 18 or older.",
  },
];

const STATUS_TEXT: Record<KeyStatus, { text: string; tone: string }> = {
  valid: { text: "Key works ✓", tone: "text-emerald-400" },
  invalid: { text: "That key was rejected — check you copied all of it.", tone: "text-red-400" },
  "rate-limited": { text: "Key looks valid, but it's rate-limited right now.", tone: "text-amber-300" },
  unreachable: { text: "Couldn't reach the service — are you online?", tone: "text-amber-300" },
};

function ProviderRow({ provider, saved }: { provider: (typeof PROVIDERS)[number]; saved?: string }) {
  const [draft, setDraft] = useState("");
  const [status, setStatus] = useState<KeyStatus | null>(null);
  const [testing, setTesting] = useState(false);
  const [warn, setWarn] = useState<string | null>(null);

  const save = () => {
    const key = draft.trim();
    if (!key) return;
    const { persisted } = setKey(provider.id, key);
    setDraft("");
    setStatus(null);
    setWarn(
      !looksLikeKey(provider.id, key)
        ? "That doesn't look like a " + provider.name + " key, but it's saved — use Test to check."
        : !persisted
          ? "Couldn't save on this device (private window?). It will be forgotten when you close this tab."
          : null,
    );
  };

  const test = async () => {
    const key = draft.trim() || saved;
    if (!key) return;
    setTesting(true);
    setStatus(null);
    try {
      setStatus(await testKey(provider.id, key, { signal: AbortSignal.timeout(10_000) }));
    } finally {
      setTesting(false);
    }
  };

  return (
    <div className="space-y-2 rounded-2xl border border-white/5 bg-black/40 p-4" data-testid={`key-row-${provider.id}`}>
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-sm font-bold text-white">{provider.name}</span>
        <a href={provider.href} target="_blank" rel="noopener noreferrer" className="text-[10px] font-bold uppercase tracking-widest text-indigo-400 hover:text-indigo-300">
          Get a free key ↗
        </a>
      </div>
      <p className="text-xs text-zinc-500">{provider.blurb}</p>

      {saved ? (
        <div className="flex items-center justify-between gap-2 rounded-xl bg-white/5 px-3 py-2">
          <span className="font-mono text-xs text-zinc-300" data-testid={`key-masked-${provider.id}`}>{maskKey(saved)}</span>
          <div className="flex gap-1">
            <Button type="button" variant="ghost" size="sm" onClick={test} disabled={testing} className="h-7 text-[10px] font-bold uppercase tracking-widest text-zinc-400 hover:text-white">
              {testing ? <Loader2 className="h-3 w-3 animate-spin" /> : "Test"}
            </Button>
            <Button type="button" variant="ghost" size="sm" onClick={() => { removeKey(provider.id); setStatus(null); setWarn(null); }} className="h-7 text-[10px] font-bold uppercase tracking-widest text-red-400 hover:text-red-300">
              Remove
            </Button>
          </div>
        </div>
      ) : (
        <div className="flex gap-2">
          <Input
            type="password"
            autoComplete="off"
            autoCapitalize="off"
            autoCorrect="off"
            spellCheck={false}
            aria-label={`${provider.name} API key`}
            placeholder={provider.placeholder}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && save()}
            className="h-9 flex-1 border-white/5 bg-black font-mono text-xs text-white"
          />
          <Button type="button" variant="ghost" onClick={test} disabled={!draft.trim() || testing} className="h-9 text-[10px] font-bold uppercase tracking-widest text-zinc-400 hover:text-white">
            {testing ? <Loader2 className="h-3 w-3 animate-spin" /> : "Test"}
          </Button>
          <Button type="button" onClick={save} disabled={!draft.trim()} className="h-9 bg-white text-[10px] font-black uppercase tracking-widest text-black hover:bg-zinc-200">
            Save
          </Button>
        </div>
      )}

      {status && <p className={`text-xs font-bold ${STATUS_TEXT[status].tone}`} data-testid={`key-status-${provider.id}`}>{STATUS_TEXT[status].text}</p>}
      {warn && <p className="text-xs text-amber-300">{warn}</p>}
      {provider.note && <p className="text-[11px] leading-relaxed text-zinc-600">{provider.note}</p>}
    </div>
  );
}

export function ApiKeySettings() {
  const [keys, setKeys] = useState<Keys>({});
  useEffect(() => {
    setKeys(getKeys());
    return onKeysChanged(() => setKeys(getKeys()));
  }, []);

  return (
    <div className="space-y-4">
      <div className="flex gap-3 rounded-2xl border border-emerald-500/20 bg-emerald-500/5 p-4 text-emerald-200">
        <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0" />
        <p className="text-xs leading-relaxed">
          Optional. Your key is stored <strong>only in this browser</strong> and is sent only to the provider you pick — never to Bill.a&apos;s servers.
          Without one, splits use on-device rules. On a shared device, remove your keys when you&apos;re done: signing out doesn&apos;t clear them.
        </p>
      </div>
      {PROVIDERS.map((p) => (
        <ProviderRow key={p.id} provider={p} saved={keys[p.id]} />
      ))}
      {Object.keys(keys).length > 1 && (
        <Button type="button" variant="ghost" onClick={() => { removeKey("groq"); removeKey("gemini"); }} className="h-8 w-full text-[10px] font-bold uppercase tracking-widest text-red-400 hover:text-red-300">
          Remove all keys from this device
        </Button>
      )}
      <p data-testid="keys-per-app-note" className="px-1 text-[11px] leading-relaxed text-zinc-600">
        Keys are saved per app, not per account. If you installed Bill.a to your home screen (on iPhone especially, where the installed
        app keeps its own storage), add your key inside that app too — a key saved in the browser won&apos;t be there.
      </p>
    </div>
  );
}
