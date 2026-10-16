// DuitNow QR (Malaysia) payloads are EMV merchant-presented QR text: a run of tag(2) length(2) value fields.
// We never create one; we only check that what a user uploads from their bank app really is a DuitNow code,
// so a random QR (a URL, a Wi-Fi code) is never forwarded to friends as "pay me here".

const DUITNOW_AID = "A0000006150001";
const CURRENCY_MYR = "458";

/** Reads the top-level tag/length/value fields; null if the text is not well-formed TLV. */
export function parseEmv(payload: string): Map<string, string> | null {
  const fields = new Map<string, string>();
  let i = 0;
  while (i < payload.length) {
    const tag = payload.slice(i, i + 2);
    const len = payload.slice(i + 2, i + 4);
    if (!/^\d{2}$/.test(tag) || !/^\d{2}$/.test(len)) return null;
    const end = i + 4 + Number(len);
    if (end > payload.length) return null;
    if (fields.has(tag)) return null;
    fields.set(tag, payload.slice(i + 4, end));
    i = end;
  }
  return fields;
}

/** EMV checksum: CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF), as 4 upper-case hex digits. */
export function crc16(text: string): string {
  let crc = 0xffff;
  for (let i = 0; i < text.length; i++) {
    crc ^= (text.charCodeAt(i) & 0xff) << 8;
    for (let b = 0; b < 8; b++) crc = crc & 0x8000 ? ((crc << 1) ^ 0x1021) & 0xffff : (crc << 1) & 0xffff;
  }
  return crc.toString(16).toUpperCase().padStart(4, "0");
}

export type DuitNowCheck = { ok: true; name?: string } | { ok: false; reason: string };

export function validateDuitNowQr(raw: string): DuitNowCheck {
  const payload = raw.trim();
  if (!payload.startsWith("000201")) return { ok: false, reason: "That isn't a payment QR code." };
  if (payload.length > 512) return { ok: false, reason: "That QR code is too long to be a DuitNow QR." };
  const fields = parseEmv(payload);
  if (!fields) return { ok: false, reason: "That QR code is damaged or incomplete." };

  // Tag 63 is the checksum and must be the last field: "6304" + 4 hex digits.
  const crc = fields.get("63");
  if (!crc || !payload.endsWith("6304" + crc) || crc.toUpperCase() !== crc16(payload.slice(0, -4))) {
    return { ok: false, reason: "That QR code's checksum doesn't match, so it may be damaged." };
  }

  // DuitNow lives in the merchant-account tags 26 to 51; the AID is sub-tag 00 of one of them.
  let isDuitNow = false;
  for (let t = 26; t <= 51; t++) {
    const inner = fields.get(String(t));
    if (inner && parseEmv(inner)?.get("00")?.toUpperCase() === DUITNOW_AID) isDuitNow = true;
  }
  if (!isDuitNow) return { ok: false, reason: "That is a payment QR, but not a DuitNow one." };
  const currency = fields.get("53");
  if (currency && currency !== CURRENCY_MYR) return { ok: false, reason: "That DuitNow QR isn't in Malaysian ringgit." };

  const name = fields.get("59")?.trim();
  return name ? { ok: true, name } : { ok: true };
}
