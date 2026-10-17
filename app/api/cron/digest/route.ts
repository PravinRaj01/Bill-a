import { timingSafeEqual } from "node:crypto";
import { NextResponse } from "next/server";

import { getDb } from "@/lib/db/client";
import { runDigest } from "@/lib/db/digest";

// Called once a day by Vercel Cron (vercel.json). Vercel sends "Authorization: Bearer <CRON_SECRET>" when that
// environment variable is set; anything without it is refused, so nobody else can trigger it.
export const dynamic = "force-dynamic";

function authorised(req: Request): boolean {
  const secret = process.env.CRON_SECRET;
  if (!secret) return false;
  const given = Buffer.from(req.headers.get("authorization") ?? "");
  const want = Buffer.from(`Bearer ${secret}`);
  return given.length === want.length && timingSafeEqual(given, want);
}

export async function GET(req: Request) {
  if (!authorised(req)) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const run = await runDigest(getDb());
  return NextResponse.json(run);
}
