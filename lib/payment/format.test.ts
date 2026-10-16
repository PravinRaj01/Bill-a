import { describe, expect, it } from "vitest";

import { hasPaymentDetails, paymentBlock, paymentLines } from "./format";

describe("paymentLines / paymentBlock", () => {
  it("is empty when nothing is set", () => {
    expect(paymentLines(null)).toEqual([]);
    expect(paymentLines({})).toEqual([]);
    expect(paymentBlock(undefined)).toBe("");
    expect(hasPaymentDetails({})).toBe(false);
  });
  it("formats each method alone", () => {
    expect(paymentLines({ duitnowId: { kind: "mobile", value: "012-3456789" } })).toEqual(["DuitNow mobile number: 012-3456789"]);
    expect(paymentLines({ bank: { bank: "Maybank", account: "1234567890", holder: "Alex Tan" } })).toEqual(["Maybank: 1234567890 (Alex Tan)"]);
    expect(paymentLines({ bank: { bank: "Maybank", account: "1234567890", holder: "" } })).toEqual(["Maybank: 1234567890"]);
    expect(paymentLines({ note: "TnG eWallet 012-3456789" })).toEqual(["TnG eWallet 012-3456789"]);
  });
  it("mentions the QR only when it will be on the image", () => {
    const d = { duitnowQr: { payload: "x" } };
    expect(paymentLines(d)).toEqual([]);
    expect(paymentLines(d, { qrAvailable: true })).toEqual(["DuitNow QR is on the image"]);
  });
  it("combines methods under a heading, in a stable order", () => {
    const block = paymentBlock({
      note: "or TnG",
      bank: { bank: "CIMB", account: "7000000000", holder: "A B" },
      duitnowId: { kind: "ic", value: "900101-01-1234" },
    });
    expect(block).toBe("Pay me:\nDuitNow ic number: 900101-01-1234\nCIMB: 7000000000 (A B)\nor TnG");
  });
});
