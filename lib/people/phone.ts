import { parsePhoneNumberFromString, type CountryCode } from "libphonenumber-js/min";

// Phone numbers are stored as E.164 ("+60123456789") so a WhatsApp link can be built from them, but people
// type them as they write them ("012-345 6789"). Malaysia is the default country, so a local number works
// without a +60; a number with its own country code ("+65 9876 5432") is honoured.

/** The number as E.164, or null when it isn't a real phone number (or is empty). */
export function normalizePhone(raw: string, country: CountryCode = "MY"): string | null {
  const text = raw.trim();
  if (!text) return null;
  const parsed = parsePhoneNumberFromString(text, country);
  return parsed && parsed.isValid() ? parsed.number : null;
}

/** An E.164 number written the way it's dialled locally: "+60123456789" -> "012-345 6789". */
export function formatPhone(e164: string): string {
  const parsed = parsePhoneNumberFromString(e164);
  if (!parsed) return e164;
  return parsed.country === "MY" ? parsed.formatNational() : parsed.formatInternational();
}

/** A WhatsApp chat with this number, with the message already typed (the number needs no "+" or spaces). */
export function waLink(e164: string, text: string): string {
  return `https://wa.me/${e164.replace(/\D/g, "")}?text=${encodeURIComponent(text)}`;
}
