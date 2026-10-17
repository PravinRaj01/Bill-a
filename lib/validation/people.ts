import { z } from "zod";

import { normalizePhone } from "@/lib/people/phone";

// A contact: a name and an optional WhatsApp number. Shown to the user only, but still bounded: the name is a
// key people are matched by, and "->" is reserved (it separates the two halves of a paid-transfer key).

const personName = z
  .string()
  .trim()
  .min(1, "Enter a name")
  .max(60, "That name is too long")
  .refine((n) => !n.includes("->"), "Names can't contain the characters \"->\"");

const phoneField = z.string().trim().max(40).nullable().optional();

export const addPersonSchema = z.object({ name: personName, phone: phoneField }).strict();
export const updatePersonSchema = z.object({ name: personName.optional(), phone: phoneField }).strict();

export type ParsedPerson = { name?: string; phone?: string | null };

/** "" or null clears a number; anything else must be a real phone number, stored as E.164. */
function phoneFrom(raw: string | null | undefined): { ok: true; value: string | null | undefined } | { ok: false; error: string } {
  if (raw === undefined) return { ok: true, value: undefined };
  if (raw === null || raw.trim() === "") return { ok: true, value: null };
  const e164 = normalizePhone(raw);
  return e164 ? { ok: true, value: e164 } : { ok: false, error: "That doesn't look like a phone number" };
}

export function parseNewPerson(input: unknown): { ok: true; value: { name: string; phone: string | null } } | { ok: false; error: string } {
  const r = addPersonSchema.safeParse(input);
  if (!r.success) return { ok: false, error: r.error.issues[0]?.message ?? "Invalid contact" };
  const phone = phoneFrom(r.data.phone);
  if (!phone.ok) return phone;
  return { ok: true, value: { name: r.data.name, phone: phone.value ?? null } };
}

export function parsePersonPatch(input: unknown): { ok: true; value: ParsedPerson } | { ok: false; error: string } {
  const r = updatePersonSchema.safeParse(input);
  if (!r.success) return { ok: false, error: r.error.issues[0]?.message ?? "Invalid contact" };
  const phone = phoneFrom(r.data.phone);
  if (!phone.ok) return phone;
  return { ok: true, value: { ...(r.data.name !== undefined ? { name: r.data.name } : {}), ...(phone.value !== undefined ? { phone: phone.value } : {}) } };
}
