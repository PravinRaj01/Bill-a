"use client";
import { useRef } from "react";
import { Image as ImageIcon, Loader2, Receipt } from "lucide-react";

import { Button } from "@/components/ui/button";
import type { SplitSession } from "@/hooks/use-split-session";

export function ScanStep({ s }: { s: SplitSession }) {
  const cameraRef = useRef<HTMLInputElement>(null);
  const galleryRef = useRef<HTMLInputElement>(null);
  const onFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const input = e.target;
    const file = input.files?.[0];
    if (!file) return;
    const ok = await s.scanFile(file);
    if (!ok) input.value = ""; // so choosing the same photo again still fires onChange
  };

  const label =
    s.scanStage === "preparing"
      ? "Preparing photo…"
      : s.scanStage === "loading-reader"
        ? "Loading reader (first time only)…"
        : "Reading receipt…";

  return (
    <div className="space-y-6">
      <div className="space-y-1">
        <h2 className="text-2xl font-bold tracking-tight text-white">Scan the receipt</h2>
        <p className="text-sm text-muted-foreground">Step 2 of 3 · read on your device, nothing is uploaded</p>
      </div>

      <div className="rounded-3xl border border-dashed border-white/10 bg-surface px-6 py-12 text-center">
        <div className="mx-auto mb-6 flex size-16 items-center justify-center rounded-full border border-white/10 bg-white/5">
          <Receipt className="size-7 text-white opacity-50" />
        </div>

        <input type="file" accept="image/*" capture="environment" ref={cameraRef} className="hidden" onChange={onFile} onClick={(e) => ((e.target as HTMLInputElement).value = "")} />
        <input type="file" accept="image/*" ref={galleryRef} className="hidden" onChange={onFile} onClick={(e) => ((e.target as HTMLInputElement).value = "")} />

        <div className="flex flex-col gap-2">
          <Button
            type="button"
            className="h-14 w-full rounded-xl bg-coral text-base font-bold text-coral-foreground shadow-[var(--coral-glow)] hover:bg-coral/90"
            onClick={() => cameraRef.current?.click()}
            disabled={s.loading}
          >
            {s.loading ? <><Loader2 className="mr-2 animate-spin" />{label}</> : "Snap photo"}
          </Button>
          <Button
            type="button"
            variant="ghost"
            className="h-12 text-sm font-semibold text-muted-foreground hover:text-white"
            onClick={() => galleryRef.current?.click()}
            disabled={s.loading}
          >
            <ImageIcon size={16} className="mr-2" /> Choose from gallery
          </Button>
        </div>
      </div>
    </div>
  );
}
