import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { config } from "dotenv";
import { inArray } from "drizzle-orm";
import { getDb } from "./client";
import { users } from "./schema";
import { getPaymentDetails, setPaymentDetails } from "./queries";
import { parsePaymentDetails } from "@/lib/validation/payment";

// Payment details live on the user row and are keyed by the session's user id. Real Neon DB (the test
// branch, never .env.local); skipped without .env.test.local. Throwaway users are removed afterwards.
config({ path: ".env.test.local", quiet: true });
const hasDb = !!process.env.DATABASE_URL;

const QR = "00020101021126410014A000000615000101065016640209123456789520400005303458540510.005802MY5909AUSERNAME6005BANGI6304F1DB";

describe.skipIf(!hasDb)("payment details (real Neon DB)", { timeout: 60_000 }, () => {
  const db = getDb();
  const tag = crypto.randomUUID().slice(0, 8);
  let A = "";
  let B = "";

  beforeAll(async () => {
    const [a, b] = await db
      .insert(users)
      .values([
        { email: `pay-a-${tag}@example.invalid`, name: "A" },
        { email: `pay-b-${tag}@example.invalid`, name: "B" },
      ])
      .returning({ id: users.id });
    A = a.id;
    B = b.id;
  });

  afterAll(async () => {
    await db.delete(users).where(inArray(users.id, [A, B].filter(Boolean)));
  });

  it("starts empty", async () => {
    expect(await getPaymentDetails(db, A)).toBeNull();
  });

  it("saves and reads back everything, scoped to the user", async () => {
    const parsed = parsePaymentDetails({
      duitnowQr: { payload: QR },
      duitnowId: { kind: "mobile", value: "012-3456789" },
      bank: { bank: "Maybank", account: "1234567890", holder: "Alex Tan" },
      note: "TnG eWallet",
    });
    if (!parsed.ok) throw new Error(parsed.error);
    await setPaymentDetails(db, A, parsed.value);
    expect(await getPaymentDetails(db, A)).toEqual(parsed.value);
    expect((await getPaymentDetails(db, A))?.duitnowQr?.name).toBe("AUSERNAME");
    // another user sees nothing of it, and cannot overwrite it
    expect(await getPaymentDetails(db, B)).toBeNull();
    await setPaymentDetails(db, B, { note: "mine" });
    expect((await getPaymentDetails(db, A))?.note).toBe("TnG eWallet");
    expect(await getPaymentDetails(db, B)).toEqual({ note: "mine" });
  });

  it("an empty object clears the details", async () => {
    expect(await setPaymentDetails(db, A, {})).toBeNull();
    expect(await getPaymentDetails(db, A)).toBeNull();
  });

  it("an unknown user id reads null and writes nothing", async () => {
    const ghost = crypto.randomUUID();
    expect(await getPaymentDetails(db, ghost)).toBeNull();
    await setPaymentDetails(db, ghost, { note: "x" });
    expect(await getPaymentDetails(db, ghost)).toBeNull();
  });
});
