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
    <main className="flex min-h-screen flex-col items-center justify-center gap-4 bg-black p-6 text-center text-white">
      <h1 className="text-lg font-bold">Something went wrong</h1>
      <p className="max-w-sm text-sm text-zinc-500">
        Your saved sessions are safe. Try again, and if it keeps happening, reload the page.
      </p>
      <Button onClick={reset} className="bg-white font-bold text-black hover:bg-zinc-200">Try again</Button>
    </main>
  );
}
