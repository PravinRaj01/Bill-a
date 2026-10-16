import { describe, expect, it } from "vitest";

import { parsePaymentDetails } from "./payment";

const QR = "00020101021126410014A000000615000101065016640209123456789520400005303458540510.005802MY5909AUSERNAME6005BANGI6304F1DB";

describe("parsePaymentDetails", () => {
  it("accepts all four methods and reads the name from the QR", () => {
    const r = parsePaymentDetails({
      duitnowQr: { payload: QR, name: "SPOOFED" },
      duitnowId: { kind: "mobile", value: " 012-3456789 " },
      bank: { bank: "Maybank", account: "1234 5678 90", holder: "Alex Tan" },
      note: "TnG eWallet",
    } as never);
    expect(r).toEqual({
      ok: true,
      value: {
        duitnowQr: { payload: QR, name: "AUSERNAME" }, // the client's name is ignored
        duitnowId: { kind: "mobile", value: "012-3456789" },
        bank: { bank: "Maybank", account: "1234 5678 90", holder: "Alex Tan" },
        note: "TnG eWallet",
      },
    });
  });
  it("derives the QR name from the payload", () => {
    const r = parsePaymentDetails({ duitnowQr: { payload: QR } });
    expect(r).toEqual({ ok: true, value: { duitnowQr: { payload: QR, name: "AUSERNAME" } } });
  });
  it("drops empty parts and accepts an empty object", () => {
    expect(parsePaymentDetails({})).toEqual({ ok: true, value: {} });
    expect(parsePaymentDetails({ duitnowId: { kind: "ic", value: "" }, note: "" })).toEqual({ ok: true, value: {} });
  });
  it("rejects a QR that is not DuitNow, with the reason", () => {
    const r = parsePaymentDetails({ duitnowQr: { payload: "https://example.com" } });
    expect(r).toEqual({ ok: false, error: "That isn't a payment QR code." });
  });
  it("rejects bad account numbers, oversized text and unknown keys", () => {
    expect(parsePaymentDetails({ bank: { bank: "X", account: "abc", holder: "" } }).ok).toBe(false);
    expect(parsePaymentDetails({ note: "x".repeat(201) }).ok).toBe(false);
    expect(parsePaymentDetails({ duitnowId: { kind: "mobile", value: "1".repeat(41) } }).ok).toBe(false);
    expect(parsePaymentDetails({ duitnowId: { kind: "email", value: "a" } }).ok).toBe(false);
    expect(parsePaymentDetails({ extra: 1 }).ok).toBe(false);
    expect(parsePaymentDetails("nope").ok).toBe(false);
  });
});
