import { z } from "zod";

import { validateDuitNowQr } from "@/lib/payment/duitnow";
import type { PaymentDetails } from "@/types/domain";

// Everything here is bounded: the details are shown to other people and stored in a jsonb column, so no
// field may be unbounded or carry markup-like surprises. Empty fields are dropped rather than stored.

const trimmed = (max: number) => z.string().trim().max(max);

export const paymentDetailsSchema = z
  .object({
    duitnowQr: z.object({ payload: z.string().trim().min(1).max(512) }).optional(),
    duitnowId: z
      .object({
        kind: z.enum(["mobile", "ic", "brn", "passport"]),
        value: trimmed(40),
      })
      .optional(),
    bank: z
      .object({
        bank: trimmed(60),
        account: z.string().trim().regex(/^[0-9][0-9 -]{4,28}[0-9]$/, "Account numbers use digits only"),
        holder: trimmed(80),
      })
      .optional(),
    note: trimmed(200).optional(),
  })
  .strict()
  .transform((d, ctx): PaymentDetails => {
    const out: PaymentDetails = {};
    if (d.duitnowQr) {
      const check = validateDuitNowQr(d.duitnowQr.payload);
      if (!check.ok) {
        ctx.addIssue({ code: "custom", message: check.reason, path: ["duitnowQr"] });
        return z.NEVER;
      }
      // The name is read back from the (checksummed) payload, never taken from the client.
      out.duitnowQr = { payload: d.duitnowQr.payload.trim(), ...(check.name ? { name: check.name } : {}) };
    }
    if (d.duitnowId?.value) out.duitnowId = { kind: d.duitnowId.kind, value: d.duitnowId.value };
    if (d.bank && d.bank.bank && d.bank.account) out.bank = { bank: d.bank.bank, account: d.bank.account, holder: d.bank.holder };
    if (d.note) out.note = d.note;
    return out;
  });

/** Parses untrusted input (a client, or rows read back from the database) into clean PaymentDetails. */
export function parsePaymentDetails(input: unknown): { ok: true; value: PaymentDetails } | { ok: false; error: string } {
  const r = paymentDetailsSchema.safeParse(input);
  return r.success ? { ok: true, value: r.data } : { ok: false, error: r.error.issues[0]?.message ?? "Invalid payment details" };
}
