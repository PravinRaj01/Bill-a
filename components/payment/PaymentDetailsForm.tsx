"use client";
import { useRef, useState } from "react";
import { Check, ImageUp, Loader2, QrCode, Trash2 } from "lucide-react";
import { QRCodeSVG } from "qrcode.react";
import { toast } from "sonner";

import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { decodeQrImage } from "@/lib/payment/decode-qr";
import { validateDuitNowQr } from "@/lib/payment/duitnow";
import { DUITNOW_ID_LABEL, paymentBlock } from "@/lib/payment/format";
import { cn } from "@/lib/utils";
import type { DuitNowIdKind, PaymentDetails } from "@/types/domain";
import { Button } from "@/components/ui/button";

const BANKS = [
  "Affin Bank", "Alliance Bank", "AmBank", "Bank Islam", "Bank Muamalat", "Bank Rakyat", "BSN", "CIMB", "Hong Leong Bank",
  "HSBC", "Maybank", "OCBC", "Public Bank", "RHB", "Standard Chartered", "UOB",
];
const OTHER = "Other";

const field = "glass-inset h-12 rounded-xl text-base text-foreground";

// Everything a friend needs to pay you back, in one editable block: a DuitNow QR read from a bank-app
// screenshot (kept as its text and redrawn crisply), a DuitNow ID, a bank account and a free note. Nothing
// is saved until Save; the preview shows exactly the lines that will be added to shared messages.
export function PaymentDetailsForm({
  initial,
  onSave,
}: {
  initial: PaymentDetails | null;
  /** Resolves to an error message, or null when saved. */
  onSave: (details: PaymentDetails) => Promise<string | null>;
}) {
  const [qr, setQr] = useState(initial?.duitnowQr ?? null);
  const [idKind, setIdKind] = useState<DuitNowIdKind>(initial?.duitnowId?.kind ?? "mobile");
  const [idValue, setIdValue] = useState(initial?.duitnowId?.value ?? "");
  const initialBank = initial?.bank?.bank ?? "";
  const [bankChoice, setBankChoice] = useState(initialBank ? (BANKS.includes(initialBank) ? initialBank : OTHER) : "");
  const [bankOther, setBankOther] = useState(initialBank && !BANKS.includes(initialBank) ? initialBank : "");
  const [account, setAccount] = useState(initial?.bank?.account ?? "");
  const [holder, setHolder] = useState(initial?.bank?.holder ?? "");
  const [note, setNote] = useState(initial?.note ?? "");
  const [qrError, setQrError] = useState<string | null>(null);
  const [reading, setReading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const bankName = bankChoice === OTHER ? bankOther.trim() : bankChoice;
  const draft: PaymentDetails = {
    ...(qr ? { duitnowQr: qr } : {}),
    ...(idValue.trim() ? { duitnowId: { kind: idKind, value: idValue.trim() } } : {}),
    ...(bankName && account.trim() ? { bank: { bank: bankName, account: account.trim(), holder: holder.trim() } } : {}),
    ...(note.trim() ? { note: note.trim() } : {}),
  };
  const preview = paymentBlock(draft, { qrAvailable: true });
  const bankHalfDone = (!!bankName || !!account.trim()) && !(bankName && account.trim());

  const pickQr = async (file: File | undefined) => {
    if (!file) return;
    setQrError(null);
    setReading(true);
    try {
      const text = await decodeQrImage(file);
      if (!text) {
        setQrError("We couldn't find a QR code in that picture. Try a screenshot of just the QR from your bank app.");
        return;
      }
      const check = validateDuitNowQr(text);
      if (!check.ok) {
        setQrError(check.reason);
        return;
      }
      setQr({ payload: text.trim(), ...(check.name ? { name: check.name } : {}) });
    } catch {
      setQrError("That file couldn't be read as a picture.");
    } finally {
      setReading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  const save = async () => {
    setError(null);
    setSaving(true);
    const message = await onSave(draft);
    setSaving(false);
    if (message) setError(message);
    else toast.success(Object.keys(draft).length ? "Payment details saved" : "Payment details cleared");
  };

  return (
    <div data-testid="payment-form" className="space-y-6 rounded-3xl glass p-5">
      {/* DuitNow QR */}
      <section className="space-y-3" aria-labelledby="pay-qr">
        <h3 id="pay-qr" className="text-sm font-bold text-foreground">
          DuitNow QR
        </h3>
        {qr ? (
          <div className="flex items-center gap-4">
            <div className="rounded-2xl bg-white p-2" data-testid="payment-qr">
              <QRCodeSVG value={qr.payload} size={132} level="M" marginSize={1} title="Your DuitNow QR" />
            </div>
            <div className="min-w-0 space-y-2">
              <p className="text-sm font-semibold text-foreground">{qr.name ?? "DuitNow QR"}</p>
              <p className="flex items-center gap-1 text-xs text-success">
                <Check className="size-3.5" strokeWidth={3} /> Checked: a real DuitNow code
              </p>
              <div className="flex flex-wrap gap-2">
                <button type="button" onClick={() => fileRef.current?.click()} className="min-h-11 rounded-xl border border-white/10 px-3 text-sm font-semibold text-foreground hover:bg-white/5">
                  Replace
                </button>
                <button
                  type="button"
                  data-testid="payment-qr-remove"
                  onClick={() => setQr(null)}
                  className="flex min-h-11 items-center gap-1.5 rounded-xl px-3 text-sm font-semibold text-muted-foreground hover:text-foreground"
                >
                  <Trash2 className="size-4" /> Remove
                </button>
              </div>
            </div>
          </div>
        ) : (
          <button
            type="button"
            data-testid="payment-qr-upload"
            onClick={() => fileRef.current?.click()}
            disabled={reading}
            className="flex min-h-14 w-full items-center gap-3 rounded-2xl border-2 border-dashed border-white/10 px-4 text-left text-sm text-muted-foreground transition-colors hover:border-white/25 hover:text-foreground disabled:opacity-60"
          >
            {reading ? <Loader2 className="size-5 shrink-0 animate-spin" /> : <ImageUp className="size-5 shrink-0" />}
            <span>
              <span className="block font-semibold text-foreground">Upload the QR from your bank app</span>
              <span className="block text-xs">A screenshot of your DuitNow QR. It&apos;s read on this device.</span>
            </span>
          </button>
        )}
        <input ref={fileRef} data-testid="payment-qr-file" type="file" accept="image/*" className="sr-only" tabIndex={-1} onChange={(e) => pickQr(e.target.files?.[0])} />
        {qrError && (
          <p role="alert" data-testid="payment-qr-error" className="text-sm text-coral">
            {qrError}
          </p>
        )}
      </section>

      {/* DuitNow ID */}
      <section className="space-y-3" aria-labelledby="pay-id">
        <h3 id="pay-id" className="text-sm font-bold text-foreground">
          DuitNow ID
        </h3>
        <div className="grid gap-3 sm:grid-cols-[11rem_1fr]">
          <div className="space-y-1.5">
            <Label htmlFor="pay-id-kind" className="text-xs text-muted-foreground">
              Type
            </Label>
            <select id="pay-id-kind" value={idKind} onChange={(e) => setIdKind(e.target.value as DuitNowIdKind)} className={cn(field, "w-full border px-3 outline-none focus-visible:ring-2 focus-visible:ring-white/30")}>
              {(Object.keys(DUITNOW_ID_LABEL) as DuitNowIdKind[]).map((k) => (
                <option key={k} value={k}>
                  {DUITNOW_ID_LABEL[k]}
                </option>
              ))}
            </select>
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="pay-id-value" className="text-xs text-muted-foreground">
              Number
            </Label>
            <Input id="pay-id-value" inputMode={idKind === "passport" ? "text" : "tel"} maxLength={40} placeholder={idKind === "mobile" ? "012-345 6789" : ""} value={idValue} onChange={(e) => setIdValue(e.target.value)} />
          </div>
        </div>
      </section>

      {/* Bank */}
      <section className="space-y-3" aria-labelledby="pay-bank">
        <h3 id="pay-bank" className="text-sm font-bold text-foreground">
          Bank account
        </h3>
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="space-y-1.5">
            <Label htmlFor="pay-bank-name" className="text-xs text-muted-foreground">
              Bank
            </Label>
            <select id="pay-bank-name" value={bankChoice} onChange={(e) => setBankChoice(e.target.value)} className={cn(field, "w-full border px-3 outline-none focus-visible:ring-2 focus-visible:ring-white/30")}>
              <option value="">Choose a bank</option>
              {BANKS.map((b) => (
                <option key={b} value={b}>
                  {b}
                </option>
              ))}
              <option value={OTHER}>{OTHER}</option>
            </select>
          </div>
          {bankChoice === OTHER && (
            <div className="space-y-1.5">
              <Label htmlFor="pay-bank-other" className="text-xs text-muted-foreground">
                Bank name
              </Label>
              <Input id="pay-bank-other" maxLength={60} value={bankOther} onChange={(e) => setBankOther(e.target.value)} />
            </div>
          )}
          <div className="space-y-1.5">
            <Label htmlFor="pay-bank-account" className="text-xs text-muted-foreground">
              Account number
            </Label>
            <Input id="pay-bank-account" inputMode="numeric" maxLength={30} value={account} onChange={(e) => setAccount(e.target.value.replace(/[^0-9 -]/g, ""))} />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="pay-bank-holder" className="text-xs text-muted-foreground">
              Name on the account
            </Label>
            <Input id="pay-bank-holder" maxLength={80} autoComplete="name" value={holder} onChange={(e) => setHolder(e.target.value)} />
          </div>
        </div>
        {bankHalfDone && <p className="text-xs text-muted-foreground">Add both the bank and the account number, or neither.</p>}
      </section>

      {/* Note */}
      <section className="space-y-1.5" aria-labelledby="pay-note">
        <Label id="pay-note" htmlFor="pay-note-input" className="text-sm font-bold text-foreground">
          Anything else
        </Label>
        <Input id="pay-note-input" maxLength={200} placeholder="e.g. TnG eWallet 012-345 6789" value={note} onChange={(e) => setNote(e.target.value)} />
      </section>

      {/* What friends see */}
      <div className="space-y-1.5 rounded-2xl bg-white/10 p-4">
        <p className="flex items-center gap-1.5 text-xs font-semibold text-muted-foreground">
          <QrCode className="size-3.5" /> This is what friends will see
        </p>
        {preview ? (
          <pre data-testid="payment-preview" className="font-sans text-sm whitespace-pre-wrap text-foreground">
            {preview}
          </pre>
        ) : (
          <p data-testid="payment-preview" className="text-sm text-muted-foreground">
            Nothing yet. Add at least one way to pay.
          </p>
        )}
      </div>

      {error && (
        <p role="alert" data-testid="payment-error" className="text-sm text-coral">
          {error}
        </p>
      )}
      <Button type="button" size="lg" data-testid="payment-save" onClick={save} disabled={saving || bankHalfDone} className="w-full">
        {saving && <Loader2 className="size-4 animate-spin" />} Save payment details
      </Button>
    </div>
  );
}
