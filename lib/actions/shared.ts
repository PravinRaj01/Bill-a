"use server";

import { z } from "zod";
import { getUserIdOrNull, requireUserId } from "@/lib/auth/session";
import { getDb } from "@/lib/db/client";
import * as s from "@/lib/db/social";
import * as q from "@/lib/db/queries";
import { paymentLines } from "@/lib/payment/format";
import { billMe } from "@/lib/split/participants";
import { localDate } from "@/lib/split/settle";
import { mySharedSummary, ownerLabel } from "@/lib/split/shared";
import type { Receipt, Settlement, SplitRecord } from "@/types/domain";

// Bills friends have shared with me (read-only), and the "I've paid" claims on them. The viewer comes from
// the session; every read goes through a bill_shares row, so a bill that isn't shared with the caller is
// simply not found.

const idSchema = z.uuid();

export interface SharedListEntry {
  id: string;
  hidden: boolean;
  billTitle: string;
  totalAmount: number;
  currency: string;
  createdAt: string;
  owner: { name: string; username: string | null; avatarSeed: string | null };
  /** What runs between me and the owner on this bill. */
  mine: { iOwe: number; theyOwe: number; settled: boolean } | null;
}

export async function listSharedWithMe(includeHidden = false): Promise<SharedListEntry[]> {
  const userId = await getUserIdOrNull();
  if (!userId) return [];
  const today = localDate();
  const rows = await s.listSharedBills(getDb(), userId, { withHidden: includeHidden });
  return rows.map((b) => {
    const m = mySharedSummary(b, today);
    return {
      id: b.id,
      hidden: b.hidden,
      billTitle: b.billTitle,
      totalAmount: b.totalAmount,
      currency: b.currency,
      createdAt: b.createdAt.toISOString(),
      owner: { name: ownerLabel(b.owner), username: b.owner.username, avatarSeed: b.owner.avatarSeed },
      mine: m ? { iOwe: m.iOwe, theyOwe: m.theyOwe, settled: m.settled } : null,
    };
  });
}

export interface SharedDetail {
  id: string;
  billTitle: string;
  totalAmount: number;
  currency: string;
  createdAt: string;
  hidden: boolean;
  owner: { name: string; username: string | null; avatarSeed: string | null };
  /** What the owner calls me on this bill, and what they call themselves. */
  asName: string;
  ownerMe: string | null;
  split: SplitRecord[];
  items: Receipt;
  settlement: Settlement | null;
  /** My open transfers with the owner, each with my claim on it if I've made one. */
  open: { key: string; amount: number; direction: "iOwe" | "theyOwe"; claim: "pending" | "rejected" | "confirmed" | null }[];
  /** The owner's payment details are only worth showing when I owe them. */
  owesOwner: boolean;
  /** How to pay the owner (text lines and their DuitNow QR), only when I owe them something on this bill. */
  payLines: string[];
  payQr: { payload: string; name?: string } | null;
}

export async function getSharedDetail(id: string): Promise<SharedDetail | null> {
  const userId = await getUserIdOrNull();
  if (!userId || !idSchema.safeParse(id).success) return null;
  const db = getDb();
  const b = await s.getSharedBill(db, userId, id);
  if (!b) return null;
  const summary = mySharedSummary(b, localDate());
  const claims = await s.listMyClaims(db, userId, id);
  const claimFor = (key: string) => claims.find((c) => c.transferKey === key)?.status ?? null;
  const owes = (summary?.iOwe ?? 0) > 0;
  const details = owes ? await q.getPaymentDetails(db, b.owner.id) : null;
  return {
    id: b.id,
    billTitle: b.billTitle,
    totalAmount: b.totalAmount,
    currency: b.currency,
    createdAt: b.createdAt.toISOString(),
    hidden: b.hidden,
    owner: { name: ownerLabel(b.owner), username: b.owner.username, avatarSeed: b.owner.avatarSeed },
    asName: b.asName,
    ownerMe: billMe(b.data) ?? null,
    split: b.data.split ?? [],
    items: b.data.items,
    settlement: b.data.settlement ?? null,
    open: (summary?.open ?? []).map((o) => ({ ...o, claim: claimFor(o.key) })),
    owesOwner: owes,
    payLines: paymentLines(details),
    payQr: details?.duitnowQr ? { payload: details.duitnowQr.payload, name: details.duitnowQr.name } : null,
  };
}

export async function hideShared(id: string, hidden: boolean): Promise<boolean> {
  const userId = await requireUserId();
  return s.setShareHidden(getDb(), userId, idSchema.parse(id), hidden);
}

export type ClaimOutcome = { ok: true } | { ok: false; error: string };

/** "I've paid": the owner confirms (marks it paid) or rejects. The amount comes from the stored bill, not from here. */
export async function claimPaid(billId: string, transferKey: string): Promise<ClaimOutcome> {
  const userId = await requireUserId();
  const r = await s.createClaim(getDb(), userId, idSchema.parse(billId), z.string().min(1).max(130).parse(transferKey));
  if (r.ok) return { ok: true };
  const messages = {
    "not-shared": "That bill isn't shared with you.",
    "no-such-transfer": "That payment isn't on the bill any more.",
    "already-paid": "That one is already marked paid.",
  } as const;
  return { ok: false, error: messages[r.reason] };
}

export interface PendingClaim {
  id: string;
  transferKey: string;
  amount: number;
  createdAt: string;
  from: { name: string; username: string | null; avatarSeed: string | null };
}

/** Friends' "I've paid" claims waiting on me, for one of my bills. */
export async function pendingClaims(billId: string): Promise<PendingClaim[]> {
  const userId = await getUserIdOrNull();
  if (!userId || !idSchema.safeParse(billId).success) return [];
  const rows = await s.listPendingClaimsForOwner(getDb(), userId, billId);
  return rows.map((c) => ({ id: c.id, transferKey: c.transferKey, amount: c.amount, createdAt: c.createdAt.toISOString(), from: { name: ownerLabel(c.from), username: c.from.username, avatarSeed: c.from.avatarSeed } }));
}

/** My bills that have a friend's "I've paid" waiting on me, with how many (for a hint on Balances). */
export async function pendingClaimsByBill(): Promise<Record<string, number>> {
  const userId = await getUserIdOrNull();
  if (!userId) return {};
  const out: Record<string, number> = {};
  for (const c of await s.listPendingClaimsForOwner(getDb(), userId)) out[c.billId] = (out[c.billId] ?? 0) + 1;
  return out;
}

export async function rejectPaidClaim(claimId: string): Promise<boolean> {
  const userId = await requireUserId();
  return s.rejectClaim(getDb(), userId, idSchema.parse(claimId));
}
