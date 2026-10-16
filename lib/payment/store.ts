import { parsePaymentDetails } from "@/lib/validation/payment";
import type { PaymentDetails } from "@/types/domain";

// Payment details for people who aren't signed in live on this device only (same shape as the account
// copy). Signed-in users use the server actions in lib/actions/user.ts instead. Storage can throw or be
// empty (private windows), so every access is guarded.

const KEY = "billa.payment.v1";

export function readLocalPaymentDetails(): PaymentDetails | null {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return null;
    const parsed = parsePaymentDetails(JSON.parse(raw));
    return parsed.ok && Object.keys(parsed.value).length ? parsed.value : null;
  } catch {
    return null;
  }
}

export function writeLocalPaymentDetails(details: PaymentDetails): void {
  try {
    if (Object.keys(details).length) localStorage.setItem(KEY, JSON.stringify(details));
    else localStorage.removeItem(KEY);
  } catch {
    /* storage unavailable: the details just aren't remembered on this device */
  }
}
