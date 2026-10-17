"use client";
import { useState } from "react";
import { Image as ImageIcon, Loader2, Plus, Trash2, TriangleAlert } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Segmented } from "@/components/ui/segmented";
import { Switch } from "@/components/ui/switch";
import { ReceiptPhoto } from "@/components/receipt-photo";
import { SplitPanel } from "@/components/split/SplitPanel";
import { centsToInput, formatMoney, parseMoneyInput } from "@/lib/money";
import type { SplitSession } from "@/hooks/use-split-session";

const inputBase = "h-11";

export function ReviewStep({ s }: { s: SplitSession }) {
  const { receipt, scanInfo, symbol } = s;
  const photoOpen = s.showPhoto && !!s.photoUrl;
  const rounding = receipt?.rounding ?? 0;
  const mine = receipt ? s.subtotal + receipt.tax + rounding : 0;

  // The rounding row shows when the receipt had one (or the Gemini re-read found one), or on request.
  const [forceRounding, setForceRounding] = useState(false);
  const [negativeDraft, setNegativeDraft] = useState(true);
  const showRounding = rounding !== 0 || forceRounding;
  const roundingNegative = rounding !== 0 ? rounding < 0 : negativeDraft;

  // Warnings worth interrupting for (the "items add up" note is reassurance, not a problem).
  const problems = scanInfo?.warnings.filter((w) => !w.startsWith("Items add up")) ?? [];
  const showWarning = !!scanInfo && (scanInfo.empty || scanInfo.confidence < 0.6 || problems.length > 0);

  const rereadBlock = (() => {
    if (!s.photo) return null;
    const diff = scanInfo?.printedTotal != null && receipt ? scanInfo.printedTotal - mine : 0;
    const unsure = !!scanInfo && (scanInfo.empty || scanInfo.confidence < 0.6 || Math.abs(diff) >= 1);
    if (s.enhanced) {
      return <p data-testid="enhanced-note" className="px-1 text-xs font-semibold text-success">✓ Re-read by Gemini — check the numbers against your receipt.</p>;
    }
    if (!unsure) {
      // A scan that looks fine still gets a quiet way to get a second opinion.
      return s.keys.gemini ? (
        <button type="button" data-testid="reread-quiet" onClick={s.enhance} disabled={s.enhancing} className="px-1 text-left text-xs text-muted-foreground hover:text-foreground/80 disabled:opacity-50">
          {s.enhancing ? "Reading…" : <>Doesn&apos;t look right? <span className="font-bold text-primary">Re-read with Gemini</span> (sends the photo to Google with your key)</>}
        </button>
      ) : (
        <p data-testid="reread-nokey" className="px-1 text-xs text-muted-foreground">
          Want a second opinion on this scan?{" "}
          <button type="button" onClick={() => s.setKeySheetOpen(true)} className="font-bold text-primary hover:text-primary">Add a free Gemini key</button>
        </p>
      );
    }
    return s.keys.gemini ? (
      <div data-testid="enhance-card" className="space-y-2 rounded-2xl border border-white/10 bg-white/[0.03] p-4">
        <p className="text-sm font-bold text-foreground">Not sure about this scan?</p>
        <p className="text-xs text-muted-foreground">Ask Gemini to re-read the photo. This sends the photo to Google using your own key.</p>
        <Button type="button" variant="ghost" onClick={s.enhance} disabled={s.enhancing} className="h-10 rounded-full border border-white/10 bg-white/5 px-4 text-xs font-bold text-foreground hover:bg-white/10 hover:text-foreground">
          {s.enhancing ? <><Loader2 className="mr-2 size-3 animate-spin" /> Reading…</> : "Re-read with Gemini"}
        </Button>
      </div>
    ) : (
      <p data-testid="enhance-hint" className="px-1 text-xs text-muted-foreground">
        Scan looks shaky.{" "}
        <button type="button" onClick={() => s.setKeySheetOpen(true)} className="font-bold text-primary hover:text-primary">Add a free Gemini key</button>{" "}
        to get a second opinion, or fix the numbers by hand.
      </p>
    );
  })();

  // Phone and tablet portrait: one column (items, then the split panel). From lg (the app's side
  // nav takes 256px, so the content needs ~1024px to fit two columns): items left, split panel
  // sticky on the right. 2xl with the photo open: photo | items | split panel.
  return (
    <div>
      {photoOpen && s.photoUrl && (
        <div className="mb-6 2xl:hidden">
          <ReceiptPhoto url={s.photoUrl} onHide={s.togglePhoto} />
        </div>
      )}
      <div className={`lg:grid lg:items-start lg:gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(320px,380px)] ${photoOpen ? "2xl:grid-cols-[minmax(0,0.9fr)_minmax(0,1fr)_minmax(340px,380px)]" : ""}`}>
      {photoOpen && s.photoUrl && (
        <div className="hidden 2xl:sticky 2xl:top-20 2xl:block">
          <ReceiptPhoto url={s.photoUrl} onHide={s.togglePhoto} />
        </div>
      )}
      <div className="space-y-6">
        {showWarning && scanInfo && (
          <div data-testid="scan-warning" className="flex gap-3 rounded-2xl border border-warning/30 bg-warning/5 p-4 text-warning">
            <TriangleAlert className="mt-0.5 size-4 shrink-0" />
            <div className="space-y-1 text-sm">
              <p className="font-bold">
                {scanInfo.empty ? "We couldn't read any items from that photo." : "Please check these numbers against your receipt."}
              </p>
              {scanInfo.empty && <p className="opacity-80">Add the items below, or go back and try a clearer photo.</p>}
              {problems.slice(0, 3).map((w, i) => (
                <p key={i} className="opacity-80">{w}</p>
              ))}
            </div>
          </div>
        )}

        {rereadBlock}

        {s.photoUrl && (
          <div className="flex justify-end">
            <Button type="button" variant="ghost" data-testid="toggle-photo" onClick={s.togglePhoto} className="h-10 rounded-full px-3 text-xs font-semibold text-muted-foreground hover:text-foreground">
              <ImageIcon size={14} className="mr-1.5" /> {s.showPhoto ? "Hide photo" : "Show photo"}
            </Button>
          </div>
        )}

        <div className="overflow-hidden rounded-3xl glass">
          <div className="border-b border-white/8 bg-white/5 px-4 py-3 text-xs font-semibold text-muted-foreground">Extracted items · tap to edit</div>
          <div className="divide-y divide-white/5">
            {receipt?.items.map((item, i) => {
              const unsure = (scanInfo?.itemConf[i] ?? 1) < 0.6;
              return (
                <div
                  key={i}
                  data-testid="review-row"
                  data-unsure={unsure ? "true" : undefined}
                  className={`flex items-center gap-2 px-4 py-3 ${unsure ? "bg-warning/5 ring-1 ring-inset ring-warning/30" : ""}`}
                >
                  <Input
                    aria-label="Item name"
                    value={item.name}
                    placeholder="Item"
                    onChange={(e) => s.editItem(i, { name: e.target.value })}
                    className={`${inputBase} min-w-0 flex-1 text-foreground`}
                  />
                  <Input
                    aria-label="Quantity"
                    key={`q${i}-${item.quantity}`}
                    type="number"
                    inputMode="numeric"
                    min={1}
                    step={1}
                    defaultValue={item.quantity}
                    onBlur={(e) => s.editItem(i, { quantity: Math.max(1, Math.round(Number(e.target.value)) || 1) })}
                    className={`${inputBase} w-14 px-1 text-center text-muted-foreground`}
                  />
                  <Input
                    aria-label="Price"
                    key={`p${i}-${item.totalPrice}`}
                    type="number"
                    inputMode="decimal"
                    min={0}
                    step={0.01}
                    defaultValue={centsToInput(item.totalPrice)}
                    onBlur={(e) => s.editItem(i, { totalPrice: parseMoneyInput(e.target.value) })}
                    className={`${inputBase} w-24 text-right font-mono tabular-nums text-foreground`}
                  />
                  <button
                    type="button"
                    aria-label="Remove item"
                    onClick={() => s.removeItem(i)}
                    className="flex size-11 shrink-0 items-center justify-center text-muted-foreground transition-colors hover:text-destructive"
                  >
                    <Trash2 className="size-4" />
                  </button>
                </div>
              );
            })}
          </div>

          {scanInfo?.printedTotal != null && receipt && receipt.items.length > 0 && (() => {
            const diff = scanInfo.printedTotal - mine;
            return diff === 0 ? (
              <p data-testid="reconcile" data-ok="true" className="px-4 pt-3 text-xs font-semibold text-success">
                ✓ Items + tax{rounding !== 0 ? " + rounding" : ""} match the receipt total ({formatMoney(scanInfo.printedTotal, symbol)})
              </p>
            ) : (
              <p data-testid="reconcile" className="px-4 pt-3 text-xs font-semibold text-warning">
                Receipt says {formatMoney(scanInfo.printedTotal, symbol)}; items + tax{rounding !== 0 ? " + rounding" : ""} come to {formatMoney(mine, symbol)} ({diff > 0 ? `${formatMoney(diff, symbol)} missing` : `${formatMoney(-diff, symbol)} too much`}).
              </p>
            );
          })()}

          <div className="px-4 pt-3">
            <Button type="button" variant="ghost" onClick={s.addItem} className="h-11 text-sm font-semibold text-muted-foreground hover:text-foreground">
              <Plus className="mr-1 size-4" /> Add item
            </Button>
          </div>

          <div className="space-y-4 p-5">
            <div className="flex items-center justify-between gap-3 rounded-xl border border-white/8 bg-white/10 p-4">
              <div className="space-y-1">
                <div className="text-sm font-medium text-white/80">Apply tax &amp; service</div>
                <div className="flex items-center gap-1 text-sm text-muted-foreground">
                  +{symbol}
                  <Input
                    aria-label="Tax and service"
                    key={`tax-${receipt?.tax}`}
                    type="number"
                    inputMode="decimal"
                    min={0}
                    step={0.01}
                    defaultValue={centsToInput(receipt?.tax ?? 0)}
                    onBlur={(e) => s.editTax(parseMoneyInput(e.target.value))}
                    className={`${inputBase} w-24 px-2 font-mono tabular-nums text-foreground/80`}
                  />
                </div>
              </div>
              <Switch checked={s.includeTax} onCheckedChange={s.setIncludeTax} aria-label="Apply tax and service" />
            </div>

            {showRounding ? (
              <div data-testid="rounding-row" className="flex flex-col gap-3 rounded-xl border border-white/8 bg-white/10 p-4 sm:flex-row sm:items-center sm:justify-between">
                <div className="space-y-1">
                  <div className="text-sm font-medium text-white/80">Rounding adjustment</div>
                  <p className="text-xs text-muted-foreground">Printed on some receipts. Follows the switch above.</p>
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <Segmented
                    label="Rounding direction"
                    className="w-28"
                    value={roundingNegative ? "neg" : "pos"}
                    onValueChange={(v) => {
                      setNegativeDraft(v === "neg");
                      if (rounding !== 0) s.editRounding(v === "neg" ? -Math.abs(rounding) : Math.abs(rounding));
                    }}
                    options={[
                      { value: "neg", label: "−" },
                      { value: "pos", label: "+" },
                    ]}
                  />
                  <Input
                    aria-label="Rounding amount"
                    key={`rounding-${rounding}`}
                    type="number"
                    inputMode="decimal"
                    min={0}
                    step={0.01}
                    defaultValue={centsToInput(Math.abs(rounding))}
                    onBlur={(e) => {
                      const abs = parseMoneyInput(e.target.value);
                      s.editRounding(roundingNegative ? -abs : abs);
                    }}
                    className={`${inputBase} w-full min-w-0 px-2 font-mono tabular-nums text-foreground/80 sm:w-24`}
                  />
                </div>
              </div>
            ) : (
              <Button type="button" variant="ghost" onClick={() => setForceRounding(true)} className="h-11 text-sm font-semibold text-muted-foreground hover:text-foreground">
                <Plus className="mr-1 size-4" /> Add rounding adjustment
              </Button>
            )}

            <div className="flex items-center justify-between border-t border-white/8 px-1 pt-4">
              <span className="text-sm font-semibold text-muted-foreground">Total to split</span>
              <span data-testid="review-total" className="text-2xl font-bold tabular-nums tracking-tight text-foreground">{formatMoney(s.displayedTotal, symbol)}</span>
            </div>
          </div>
        </div>
      </div>
      <div className="mt-6 lg:sticky lg:top-20 lg:mt-0">
        <SplitPanel key={s.people.join("|")} s={s} />
      </div>
      </div>
    </div>
  );
}
