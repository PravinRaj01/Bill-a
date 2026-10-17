"use client";
import Link from "next/link";

import { Switch } from "@/components/ui/switch";
import type { PaymentShare } from "@/hooks/use-payment-share";

// "Add my payment details" on the share buttons. Only shown once the user has saved some, so it never
// nags; the choice is remembered on this device.
export function PaymentToggle({ payment }: { payment: PaymentShare }) {
  if (!payment.details) return null;
  return (
    <label data-testid="payment-toggle" className="flex min-h-11 items-center justify-between gap-3 px-1 text-sm">
      <span className="text-muted-foreground">
        Add my payment details to messages{" "}
        <Link href="/dashboard/account" className="font-semibold text-foreground underline underline-offset-4">
          Edit
        </Link>
      </span>
      <Switch checked={payment.include} onCheckedChange={payment.setInclude} aria-label="Add my payment details to messages" />
    </label>
  );
}
