import { forwardRef } from "react";
import { QRCodeSVG } from "qrcode.react";

import { formatMoney } from "@/lib/money";
import { paymentLines } from "@/lib/payment/format";
import { parseItems } from "@/lib/split/items";
import { positionNote, positions, shortDate, type SettleView } from "@/lib/split/settle";
import { initials, personColor } from "@/lib/ui/person-color";
import { SYSTEM_FONT, SYSTEM_MONO } from "@/lib/ui/share-image";
import type { PaymentDetails, Settlement, SplitRecord } from "@/types/domain";

export const RECEIPT_IMAGE_WIDTH = 1080;

export interface ReceiptImageData {
  title: string;
  date: string | number | Date;
  symbol: string;
  shares: SplitRecord[];
  settlement: Settlement | null;
  view: SettleView | null;
  /** Shown at the bottom as "Pay <me>"; null leaves the pay block out. */
  payment: PaymentDetails | null;
  /** Portrait data URIs by person name (lib/ui/avatar.ts); a person without one gets their initials. */
  avatars?: Record<string, string>;
}

// The receipt as one tall picture, sized for a phone chat. It is a plain static tree on purpose: no
// animated NumberFlow (it draws in a shadow root, which screenshot libraries capture unreliably), no
// hover or motion, amounts as text. Every number comes from the same data as the on-screen ticket.
export const ReceiptImage = forwardRef<HTMLDivElement, { data: ReceiptImageData; systemFont?: boolean }>(function ReceiptImage({ data, systemFont }, ref) {
  const { title, date, symbol, shares, settlement, view, payment, avatars } = data;
  const money = (c: number) => formatMoney(c, symbol);
  const total = shares.reduce((a, s) => a + s.amount, 0);
  const positionOf = new Map(view?.ok && settlement ? positions(shares, settlement.paidBy).map((p) => [p.name, p] as const) : []);
  const payers = (settlement?.paidBy ?? []).filter((p) => p.amount > 0);
  const open = view?.ok ? view.transfers.filter((t) => !t.paid) : [];
  const lines = paymentLines(payment);
  const me = settlement?.me;

  return (
    <div
      ref={ref}
      data-testid="receipt-image"
      style={{
        width: RECEIPT_IMAGE_WIDTH,
        fontFamily: systemFont ? SYSTEM_FONT : "var(--font-jakarta), system-ui, sans-serif",
        ...(systemFont ? ({ "--font-mono": SYSTEM_MONO, "--font-geist-mono": SYSTEM_MONO } as React.CSSProperties) : {}),
      }}
      className="bg-background p-14 text-white"
    >
      <div className="rounded-[48px] bg-surface p-12">
        <div className="flex items-center justify-between">
          <span className="rounded-full bg-coral/15 px-6 py-2 text-3xl font-bold tracking-widest text-coral uppercase">Receipt</span>
          <span className="font-mono text-3xl tracking-wider text-muted-foreground uppercase">{shortDate(date)}</span>
        </div>
        <h1 className="mt-8 text-7xl leading-tight font-black tracking-tight break-words">{title}</h1>
        <p className="mt-8 font-mono text-3xl tracking-widest text-muted-foreground uppercase">Total</p>
        <p className="text-8xl font-black tracking-tight">{money(total)}</p>
        {shares.length > 1 && <p className="mt-2 text-3xl text-muted-foreground">Split {shares.length} ways</p>}

        <div className="my-10 border-t-4 border-dashed border-white/15" />

        <ul className="space-y-8">
          {shares.map((s) => {
            const chips = parseItems(s.items);
            const pos = positionOf.get(s.name);
            const note = pos ? positionNote(pos, symbol) : "";
            return (
              <li key={s.name} className="flex items-start gap-6">
                <span
                  className="flex size-20 shrink-0 items-center justify-center overflow-hidden rounded-full bg-surface-2 text-3xl font-bold"
                  style={{ boxShadow: `0 0 0 4px ${personColor(s.name)}` }}
                >
                  {avatars?.[s.name] ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={avatars[s.name]} alt="" width={80} height={80} className="size-full" />
                  ) : (
                    initials(s.name)
                  )}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex items-baseline justify-between gap-6">
                    <span className="text-5xl font-bold break-words">{s.name}</span>
                    <span className="shrink-0 text-5xl font-bold tabular-nums">{money(s.amount)}</span>
                  </div>
                  {note && <p className="mt-1 text-3xl text-muted-foreground">{note}</p>}
                  {chips.length > 0 && (
                    <p className="mt-2 text-3xl leading-snug text-zinc-400">{chips.map((c) => (c.share ? `${c.name} ${c.share}` : c.name)).join(" · ")}</p>
                  )}
                </div>
              </li>
            );
          })}
        </ul>

        {(payers.length > 0 || view?.ok) && (
          <>
            <div className="my-10 border-t-4 border-dashed border-white/15" />
            <div className="space-y-4 text-4xl">
              {payers.length > 0 && (
                <p className="text-muted-foreground">
                  Paid by <span className="font-bold text-white">{payers.map((p) => (payers.length > 1 ? `${p.name} (${money(p.amount)})` : p.name)).join(", ")}</span>
                </p>
              )}
              {view?.ok && open.length > 0 && (
                <>
                  <p className="font-mono text-3xl tracking-widest text-muted-foreground uppercase">Still to pay</p>
                  <ul className="space-y-3">
                    {open.map((t) => (
                      <li key={t.key} className="flex items-baseline justify-between gap-6">
                        <span className="font-semibold">
                          {t.from} <span className="text-muted-foreground">→</span> {t.to}
                        </span>
                        <span className="font-bold tabular-nums">{money(t.amount)}</span>
                      </li>
                    ))}
                  </ul>
                </>
              )}
              {view?.ok && view.transfers.length > 0 && open.length === 0 && <p className="font-bold text-success">All settled ✓</p>}
            </div>
          </>
        )}

        {payment && (payment.duitnowQr || lines.length > 0) && (
          <>
            <div className="my-10 border-t-4 border-dashed border-white/15" />
            <p className="text-5xl font-black">Pay {me ?? "me"}</p>
            <div className="mt-8 flex items-center gap-10">
              {payment.duitnowQr && (
                <div className="shrink-0 rounded-3xl bg-white p-5">
                  <QRCodeSVG value={payment.duitnowQr.payload} size={360} level="M" marginSize={0} />
                  <p className="mt-3 text-center text-3xl font-bold text-black">DuitNow</p>
                </div>
              )}
              <div className="min-w-0 flex-1 space-y-4 text-4xl leading-snug">
                {payment.duitnowQr?.name && <p className="text-muted-foreground">{payment.duitnowQr.name}</p>}
                {lines.map((l) => (
                  <p key={l} className="font-semibold break-words">
                    {l}
                  </p>
                ))}
                {payment.duitnowQr && <p className="text-3xl text-muted-foreground">Scan with any banking app</p>}
              </div>
            </div>
          </>
        )}

        <p className="mt-12 text-center font-mono text-2xl tracking-[0.3em] text-muted-foreground uppercase">Split with Bill.a</p>
      </div>
    </div>
  );
});
