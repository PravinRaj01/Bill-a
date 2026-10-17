"use client";

import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import { trackError } from "@/lib/telemetry/client";

// Route-level error boundary: a render crash shows a calm message with a retry instead
// of a blank screen, and reports the (redacted, key-free) error name for the logs.
export default function RouteError({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  useEffect(() => {
    trackError("render", error);
  }, [error]);

  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-4 p-6 text-center text-foreground">
      <h1 className="text-lg font-bold">Something went wrong</h1>
      <p className="max-w-sm text-sm text-muted-foreground">
        Your saved sessions are safe. Try again, and if it keeps happening, reload the page.
      </p>
      <Button onClick={reset}>Try again</Button>
    </main>
  );
}
