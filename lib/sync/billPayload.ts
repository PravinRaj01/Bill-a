import type { BillHistoryRow } from "@/lib/db/schema";
import type { BillPayload } from "@/lib/validation/bill";
import type { Participant, Settlement } from "@/types/domain";

/**
 * The save payload for an already-saved bill with a different settlement (null removes it). It keeps the
 * bill's own clientId, so the server row is UPDATED rather than duplicated, and it goes through the
 * local-first outbox like every other save.
 */
export function payloadWithSettlement(bill: BillHistoryRow, settlement: Settlement | null): BillPayload {
  const data = { ...bill.data };
  delete data.settlement;
  const tier = bill.providerTier === "groq" || bill.providerTier === "gemini" || bill.providerTier === "fallback" ? bill.providerTier : null;
  return {
    clientId: bill.clientId,
    billTitle: bill.billTitle,
    merchantCategory: bill.merchantCategory,
    totalAmount: bill.totalAmount,
    currency: bill.currency,
    providerTier: tier,
    data: settlement ? { ...data, settlement } : data,
  };
}

/** The save payload for an already-saved bill with different participants (who is linked to a friend's account). */
export function payloadWithParticipants(bill: BillHistoryRow, participants: Participant[]): BillPayload {
  return { ...payloadWithSettlement(bill, bill.data.settlement ?? null), data: { ...bill.data, participants } };
}
