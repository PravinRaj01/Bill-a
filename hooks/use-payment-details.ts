"use client";
import { useCallback, useEffect, useState } from "react";

import { getPaymentDetails, savePaymentDetails } from "@/lib/actions/user";
import { readLocalPaymentDetails, writeLocalPaymentDetails } from "@/lib/payment/store";
import { parsePaymentDetails } from "@/lib/validation/payment";
import type { PaymentDetails } from "@/types/domain";

// The user's payment details: from the account when signed in, from this device when not. `details` is
// undefined until loaded, then the saved details or null.
export function usePaymentDetails(signedIn: boolean | undefined) {
  const [details, setDetails] = useState<PaymentDetails | null | undefined>(undefined);

  useEffect(() => {
    if (signedIn === undefined) return;
    let cancelled = false;
    const load = signedIn ? getPaymentDetails().catch(() => null) : Promise.resolve(readLocalPaymentDetails());
    load.then((d) => !cancelled && setDetails(d));
    return () => {
      cancelled = true;
    };
  }, [signedIn]);

  /** Saves (an empty object clears). Resolves to an error message, or null on success. */
  const save = useCallback(
    async (next: PaymentDetails): Promise<string | null> => {
      if (signedIn) {
        try {
          const r = await savePaymentDetails(next);
          if (!r.ok) return r.error;
          setDetails(r.details);
          return null;
        } catch {
          return "Couldn't save. Check your connection and try again.";
        }
      }
      const parsed = parsePaymentDetails(next);
      if (!parsed.ok) return parsed.error;
      writeLocalPaymentDetails(parsed.value);
      setDetails(Object.keys(parsed.value).length ? parsed.value : null);
      return null;
    },
    [signedIn],
  );

  return { details, save };
}
