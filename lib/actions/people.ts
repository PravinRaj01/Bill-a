"use server";

import { z } from "zod";
import { getUserIdOrNull, requireUserId } from "@/lib/auth/session";
import { auth } from "@/lib/auth";
import { getDb } from "@/lib/db/client";
import * as q from "@/lib/db/queries";
import { meNameFor } from "@/lib/split/participants";
import { parseNewPerson, parsePersonPatch } from "@/lib/validation/people";

// The user's contacts. Server actions are public endpoints: every argument is validated, and the user's
// identity comes from the session only (lib/auth/session.ts), never from an argument.

const idSchema = z.uuid();

/** A contact; `friend` is the public side of the Bill.a account it is linked to (only on the list, absent after an edit). */
export type PersonView = Awaited<ReturnType<typeof q.listPeople>>[number] & { friend?: { name: string | null; username: string | null; avatarSeed: string | null } | null };

/** Guests have no contacts, so they get an empty list rather than an error. */
export async function listPeople(): Promise<PersonView[]> {
  const userId = await getUserIdOrNull();
  return userId ? q.listPeopleWithFriends(getDb(), userId) : [];
}

export type SavePersonResult = { ok: true; person: PersonView } | { ok: false; error: string };

/** Adds a contact (or updates the one with the same name: case and spacing don't make a new person). */
export async function addPerson(input: unknown): Promise<SavePersonResult> {
  const userId = await requireUserId();
  const parsed = parseNewPerson(input);
  if (!parsed.ok) return { ok: false, error: parsed.error };
  return { ok: true, person: await q.upsertPerson(getDb(), userId, parsed.value) };
}

/** Renames a contact and/or sets or clears their number. */
export async function updatePerson(id: string, input: unknown): Promise<SavePersonResult> {
  const userId = await requireUserId();
  const parsed = parsePersonPatch(input);
  if (!parsed.ok || !idSchema.safeParse(id).success) return { ok: false, error: parsed.ok ? "Couldn't find that person." : parsed.error };
  const r = await q.updatePerson(getDb(), userId, id, parsed.value);
  if (r.ok) return { ok: true, person: r.person };
  return { ok: false, error: r.reason === "name-taken" ? "You already have someone with that name." : "Couldn't find that person." };
}

export async function removePerson(id: string): Promise<boolean> {
  const userId = await requireUserId();
  return q.deletePerson(getDb(), userId, idSchema.parse(id));
}

/** Names from your bills and groups that aren't contacts yet (your own name is left out). */
export async function suggestPeople() {
  const userId = await getUserIdOrNull();
  if (!userId) return [];
  const me = meNameFor((await auth())?.user?.name);
  return q.suggestPeople(getDb(), userId, [me]);
}
