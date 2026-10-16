import type { DuitNowIdKind, PaymentDetails } from "@/types/domain";

export const DUITNOW_ID_LABEL: Record<DuitNowIdKind, string> = {
  mobile: "Mobile number",
  ic: "IC number",
  brn: "Business reg. no.",
  passport: "Passport number",
};

/** True when there is at least one way to pay in `details`. */
export function hasPaymentDetails(details: PaymentDetails | null | undefined): details is PaymentDetails {
  return !!details && !!(details.duitnowQr || details.duitnowId || details.bank || details.note);
}

/**
 * The lines a friend needs to pay, shared by the text messages and the receipt image so they always agree.
 * The DuitNow QR itself can't be text; it is drawn on the image, so here it is only a hint to look for it.
 */
export function paymentLines(details: PaymentDetails | null | undefined, opts: { qrAvailable?: boolean } = {}): string[] {
  if (!hasPaymentDetails(details)) return [];
  const lines: string[] = [];
  if (details.duitnowId) lines.push(`DuitNow ${DUITNOW_ID_LABEL[details.duitnowId.kind].toLowerCase()}: ${details.duitnowId.value}`);
  if (details.bank) {
    const { bank, account, holder } = details.bank;
    lines.push(`${bank}: ${account}${holder ? ` (${holder})` : ""}`);
  }
  if (details.note) lines.push(details.note);
  if (details.duitnowQr && opts.qrAvailable) lines.push("DuitNow QR is on the image");
  return lines;
}

/** A "Pay me:" block to append to a message, or "" when there is nothing to show. */
export function paymentBlock(details: PaymentDetails | null | undefined, opts: { qrAvailable?: boolean } = {}): string {
  const lines = paymentLines(details, opts);
  return lines.length ? ["Pay me:", ...lines].join("\n") : "";
}
