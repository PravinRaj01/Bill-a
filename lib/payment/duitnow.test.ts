import { describe, expect, it } from "vitest";

import { crc16, parseEmv, validateDuitNowQr } from "./duitnow";

// PayNet's published example of a DuitNow P2P static QR (MCC 0000, 10.00 MYR). PayNet's page prints the
// format indicator as "02"; real bank codes use "01" (000201), so the fixture uses "01" and its checksum
// is recomputed. The original string is kept to prove the checksum routine itself matches PayNet's.
const PAYNET_AS_PRINTED = "00020201021126410014A000000615000101065016640209123456789520400005303458540510.005802MY5909AUSERNAME6005BANGI63043A23";
const P2P = "00020101021126410014A000000615000101065016640209123456789520400005303458540510.005802MY5909AUSERNAME6005BANGI6304F1DB";

/** Rebuild a payload with a valid checksum after editing its body. */
const withCrc = (body: string) => body + crc16(body);

describe("crc16", () => {
  it("matches PayNet's example checksum", () => {
    expect(crc16(PAYNET_AS_PRINTED.slice(0, -4))).toBe("3A23");
    expect(crc16(P2P.slice(0, -4))).toBe("F1DB");
  });
  it("pads to four digits", () => {
    expect(crc16("")).toBe("FFFF");
    expect(crc16("x")).toHaveLength(4);
  });
});

describe("parseEmv", () => {
  it("reads the top-level fields", () => {
    const f = parseEmv(P2P)!;
    expect(f.get("01")).toBe("11");
    expect(f.get("59")).toBe("AUSERNAME");
    expect(f.get("63")).toBe("F1DB");
  });
  it("rejects a field that runs past the end, and a repeated tag", () => {
    expect(parseEmv("0005AB")).toBeNull();
    expect(parseEmv("000201000201")).toBeNull();
    expect(parseEmv("hello world")).toBeNull();
  });
});

describe("validateDuitNowQr", () => {
  it("accepts PayNet's example and returns the account name", () => {
    expect(validateDuitNowQr(P2P)).toEqual({ ok: true, name: "AUSERNAME" });
    expect(validateDuitNowQr(`  ${P2P}\n`).ok).toBe(true);
  });
  it("accepts a code without a name", () => {
    const body = "00020101021126410014A00000061500010106501664020912345678952040000530345858" + "02MY" + "6304";
    expect(validateDuitNowQr(withCrc(body))).toEqual({ ok: true });
  });
  it("rejects a one-character change", () => {
    const tampered = P2P.replace("AUSERNAME", "BUSERNAME");
    const r = validateDuitNowQr(tampered);
    expect(r.ok).toBe(false);
  });
  it("rejects a valid EMV code that isn't DuitNow", () => {
    const body = P2P.slice(0, -4).replace("A0000006150001", "A0000006150002");
    const r = validateDuitNowQr(withCrc(body));
    expect(r).toEqual({ ok: false, reason: "That is a payment QR, but not a DuitNow one." });
  });
  it("rejects other currencies", () => {
    const body = P2P.slice(0, -4).replace("5303458", "5303840");
    expect(validateDuitNowQr(withCrc(body)).ok).toBe(false);
  });
  it("rejects non-payment QR text", () => {
    expect(validateDuitNowQr("https://example.com/pay")).toEqual({ ok: false, reason: "That isn't a payment QR code." });
    expect(validateDuitNowQr("")).toMatchObject({ ok: false });
    expect(validateDuitNowQr("000201" + "x".repeat(600))).toMatchObject({ ok: false });
  });
  it("rejects a missing checksum", () => {
    expect(validateDuitNowQr(P2P.slice(0, -8)).ok).toBe(false);
  });
});
