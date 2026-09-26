import NextAuth from "next-auth";
import { authConfig } from "@/lib/auth.config";

// Uses the EDGE-SAFE config only (no adapter, no argon2, no database). With
// JWT sessions this just verifies the signed cookie, so middleware never
// touches Neon — a cold database can't slow down or break navigation.
// The protection rules themselves live in authConfig.callbacks.authorized.
const { auth } = NextAuth(authConfig);
export default auth;

export const config = {
  matcher: [
    // /api/auth is Auth.js's own endpoints. /dev/* is local diagnostic
    // tooling with no business being auth-gated.
    "/((?!_next/static|_next/image|favicon.ico|api/auth|dev).*)",
  ],
};
