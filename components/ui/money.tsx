"use client";
import NumberFlow from "@number-flow/react";

import { currencyPrefix } from "@/lib/money";
import type { Cents } from "@/types/domain";

// A money amount whose digits roll when it changes (respects prefers-reduced-motion by
// default). The currency symbol is the app's own ("RM", "$"), not Intl's, so it matches
// formatMoney() everywhere else; digits are tabular so columns of amounts line up.
export function Money({
  cents,
  currency = "RM",
  className,
  signed = false,
}: {
  cents: Cents;
  currency?: string;
  className?: string;
  /** Show an explicit + for positive amounts (for rounding deltas). */
  signed?: boolean;
}) {
  return (
    <NumberFlow
      value={cents / 100}
      prefix={currencyPrefix(currency)}
      format={{ minimumFractionDigits: 2, maximumFractionDigits: 2, signDisplay: signed ? "exceptZero" : "auto" }}
      locales="en-US"
      className={className}
      style={{ fontVariantNumeric: "tabular-nums" }}
      data-cents={cents}
    />
  );
}
