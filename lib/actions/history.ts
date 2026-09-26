"use server";

import { z } from "zod";
import { getDb } from "@/lib/db/client";
import * as q from "@/lib/db/queries";
import { getUserIdOrNull, requireUserId } from "@/lib/auth/session";

const idSchema = z.uuid();

// Cents are integers everywhere from here down. `data` is bounded so a client
// can't stash an arbitrarily large blob in our database.
const billSchema = z.object({
  clientId: z.uuid(),
  billTitle: z.string().trim().min(1).max(120),
  merchantCategory: z.string().trim().max(40).nullish(),
  totalAmount: z.number().int().min(-100_000_000).max(100_000_000),
  currency: z.string().trim().min(1).max(8),
  providerTier: z.enum(["groq", "gemini", "fallback"]).nullish(),
  data: z.object({
    split: z
      .array(z.object({ name: z.string().max(60), amount: z.number().int(), items: z.string().max(2000) }))
      .max(50),
    items: z.object({
      items: z
        .array(
          z.object({
            name: z.string().max(200),
            quantity: z.number(),
            unitPrice: z.number().int(),
            totalPrice: z.number().int(),
          }),
        )
        .max(300),
      tax: z.number().int(),
      total: z.number().int(),
      currency: z.string().max(8),
    }),
    people: z.array(z.string().max(60)).max(50),
    reasoning: z.string().max(20_000),
  }),
});

export async function listBills() {
  const userId = await getUserIdOrNull();
  return userId ? q.listBills(getDb(), userId) : [];
}

export async function getBill(id: string) {
  const userId = await getUserIdOrNull();
  if (!userId || !idSchema.safeParse(id).success) return null;
  return q.getBill(getDb(), userId, id);
}

export async function saveBill(input: z.input<typeof billSchema>) {
  const userId = await requireUserId();
  return q.upsertBill(getDb(), userId, billSchema.parse(input));
}

export async function deleteBills(ids: string[]) {
  const userId = await requireUserId();
  return q.deleteBills(getDb(), userId, z.array(idSchema).max(500).parse(ids));
}

export async function clearHistory() {
  const userId = await requireUserId();
  return q.clearBills(getDb(), userId);
}

export async function getAccountStats() {
  const userId = await getUserIdOrNull();
  return userId ? q.billStats(getDb(), userId) : { count: 0, totalCents: 0 };
}

/** "Session N" numbering: computed server-side from the caller's own count. */
export async function nextSessionTitle() {
  const userId = await getUserIdOrNull();
  if (!userId) return "Session 1";
  const { count } = await q.billStats(getDb(), userId);
  return `Session ${count + 1}`;
}
