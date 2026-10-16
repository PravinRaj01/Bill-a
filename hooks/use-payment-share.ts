"use client";
import { useCallback, useEffect, useState } from "react";

import { usePaymentDetails } from "@/hooks/use-payment-details";
import { getCurrentUser } from "@/lib/actions/user";
import { hasPaymentDetails, paymentBlock } from "@/lib/payment/format";
import type { PaymentDetails } from "@/types/domain";

// What the share buttons need to know about the user's payment details: do they exist, are they switched
// on for sharing (remembered on this device), and the text block to append. `block` is "" when there is
// nothing to add, so callers can pass it straight through.

export interface PaymentShare {
  /** undefined until we know; false for guests. */
  signedIn: boolean | undefined;
  details: PaymentDetails | null;
  include: boolean;
  setInclude: (on: boolean) => void;
  block: string;
}

const KEY = "billa.payment.include";

// Nothing rendered depends on this until the details have loaded, so reading it up front can't mismatch the server render.
function readInclude(): boolean {
  try {
    return typeof localStorage === "undefined" || localStorage.getItem(KEY) !== "0";
  } catch {
    return true;
  }
}

export function usePaymentShare(): PaymentShare {
  const [signedIn, setSignedIn] = useState<boolean | undefined>(undefined);
  const [include, setIncludeState] = useState(readInclude);

  useEffect(() => {
    getCurrentUser().then((u) => setSignedIn(!!u)).catch(() => setSignedIn(false));
  }, []);

  const { details } = usePaymentDetails(signedIn);
  const setInclude = useCallback((on: boolean) => {
    setIncludeState(on);
    try {
      localStorage.setItem(KEY, on ? "1" : "0");
    } catch {
      /* not remembered */
    }
  }, []);

  const saved = details && hasPaymentDetails(details) ? details : null;
  return { signedIn, details: saved, include, setInclude, block: saved && include ? paymentBlock(saved) : "" };
}
