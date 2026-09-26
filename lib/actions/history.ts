"use server";

import { z } from "zod";
import { getDb } from "@/lib/db/client";
import * as q from "@/lib/db/queries";
import { getUserIdOrNull, requireUserId } from "@/lib/auth/session";

const idSchema = z.uuid();

export async function listBills() {
  const userId = await getUserIdOrNull();
  return userId ? q.listBills(getDb(), userId) : [];
}

export async function getBill(id: string) {
  const userId = await getUserIdOrNull();
  if (!userId || !idSchema.safeParse(id).success) return null;
  return q.getBill(getDb(), userId, id);
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
