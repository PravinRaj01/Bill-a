import { telemetryEventSchema } from "@/lib/telemetry/events";

// Receives the anonymous events from lib/telemetry/client.ts and writes them to the
// platform logs (Vercel → Logs), one JSON line each. No database, no third party.
//
// The schema is strict: anything outside the allowlist is rejected, so even a modified
// or malicious client cannot make us store keys, names or prices. The request's IP and
// user agent are deliberately not logged by this code (the hosting platform keeps its
// own access logs, which we don't control).

export const runtime = "nodejs";

const MAX_BYTES = 2048;

export async function POST(req: Request) {
  const text = await req.text().catch(() => "");
  if (text.length === 0 || text.length > MAX_BYTES) return new Response(null, { status: 400 });

  let json: unknown;
  try {
    json = JSON.parse(text);
  } catch {
    return new Response(null, { status: 400 });
  }

  const parsed = telemetryEventSchema.safeParse(json);
  // A generic 400: never echo the rejected payload back.
  if (!parsed.success) return new Response(null, { status: 400 });

  console.log(JSON.stringify({ telemetry: parsed.data }));
  return new Response(null, { status: 204 });
}
