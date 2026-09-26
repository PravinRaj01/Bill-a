"use client";

import { useEffect, useState } from "react";
import { X, ZoomIn, ZoomOut } from "lucide-react";
import { Button } from "@/components/ui/button";

/** An object URL for a Blob, revoked when the blob changes or the component unmounts. */
export function useObjectUrl(blob: Blob | null): string | null {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!blob) {
      setUrl(null);
      return;
    }
    const u = URL.createObjectURL(blob);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [blob]);
  return url;
}

/**
 * The scanned receipt, shown next to (large screens) or above (phones) the extracted items,
 * so numbers can be checked against the paper. Tap it to enlarge. The full-screen viewer has
 * its own zoom control because the app disables page pinch-zoom (it prevents accidental zoom
 * when focusing inputs), which would otherwise leave phone users unable to read small print.
 */
export function ReceiptPhoto({ url, onHide }: { url: string; onHide: () => void }) {
  const [open, setOpen] = useState(false);
  const [zoomed, setZoomed] = useState(false);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  return (
    <>
      <div data-testid="receipt-photo" className="overflow-hidden rounded-3xl border border-white/10 bg-[#0c0c0e]">
        <div className="flex items-center justify-between border-b border-white/5 bg-white/5 px-4 py-3">
          <span className="text-[10px] font-bold uppercase tracking-widest text-slate-500">Your receipt · tap to enlarge</span>
          <button type="button" aria-label="Hide photo" onClick={onHide} className="text-zinc-500 hover:text-white">
            <X className="h-4 w-4" />
          </button>
        </div>
        <button type="button" onClick={() => { setZoomed(false); setOpen(true); }} className="block w-full cursor-zoom-in bg-black p-2">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={url} alt="The receipt you scanned" className="mx-auto max-h-[45vh] w-auto max-w-full object-contain lg:max-h-[78vh]" />
        </button>
      </div>

      {open && (
        <div role="dialog" aria-modal="true" aria-label="Receipt photo" data-testid="receipt-photo-viewer" className="fixed inset-0 z-[100] bg-black/95">
          <div className="absolute right-3 top-3 z-10 flex gap-2">
            <Button type="button" variant="ghost" aria-label={zoomed ? "Fit to screen" : "Zoom in"} onClick={() => setZoomed((z) => !z)} className="h-10 rounded-full bg-white/10 px-3 text-white hover:bg-white/20 hover:text-white">
              {zoomed ? <ZoomOut className="h-4 w-4" /> : <ZoomIn className="h-4 w-4" />}
            </Button>
            <Button type="button" variant="ghost" aria-label="Close" onClick={() => setOpen(false)} className="h-10 rounded-full bg-white/10 px-3 text-white hover:bg-white/20 hover:text-white">
              <X className="h-4 w-4" />
            </Button>
          </div>
          <div className="h-full w-full overflow-auto p-2" onClick={() => !zoomed && setOpen(false)}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={url}
              alt="The receipt you scanned, enlarged"
              onClick={(e) => { e.stopPropagation(); setZoomed((z) => !z); }}
              className={zoomed ? "max-w-none cursor-zoom-out" : "mx-auto max-h-full max-w-full cursor-zoom-in object-contain"}
              style={zoomed ? { width: "220%" } : undefined}
            />
          </div>
        </div>
      )}
    </>
  );
}
