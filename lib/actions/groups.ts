"use server";

import { z } from "zod";
import { getDb } from "@/lib/db/client";
import * as q from "@/lib/db/queries";
import { getUserIdOrNull, requireUserId } from "@/lib/auth/session";

// Server actions are public endpoints: validate every argument, and take the
// user's identity from the session only (lib/auth/session.ts).

const idSchema = z.uuid();
const namesSchema = z
  .array(z.string().trim().min(1).max(60))
  .min(1)
  .max(50);
const groupNameSchema = z.string().trim().min(1).max(60);

/** Guests have no saved groups, so they get an empty list rather than an error. */
export async function listGroups() {
  const userId = await getUserIdOrNull();
  return userId ? q.listGroups(getDb(), userId) : [];
}

export async function getGroup(id: string) {
  const userId = await getUserIdOrNull();
  if (!userId || !idSchema.safeParse(id).success) return null;
  return q.getGroup(getDb(), userId, id);
}

export async function saveGroup(input: { groupName: string; names: string[] }) {
  const userId = await requireUserId();
  const parsed = z.object({ groupName: groupNameSchema, names: namesSchema }).parse(input);
  return q.createGroup(getDb(), userId, parsed);
}

export async function updateGroupNames(id: string, names: string[]) {
  const userId = await requireUserId();
  return q.updateGroupNames(getDb(), userId, idSchema.parse(id), namesSchema.parse(names));
}

export async function deleteGroup(id: string) {
  const userId = await requireUserId();
  return q.deleteGroup(getDb(), userId, idSchema.parse(id));
}
