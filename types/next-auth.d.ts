import type { DefaultSession } from "next-auth";
import "next-auth/jwt";

// session.user.id is always present once signed in: lib/auth.config.ts copies
// it from the JWT's `sub`. Every server-side query is scoped with it.
declare module "next-auth" {
  interface Session {
    user: { id: string; username?: string | null; avatarSeed?: string | null } & DefaultSession["user"];
  }
}

declare module "next-auth/jwt" {
  interface JWT {
    username?: string | null;
    avatarSeed?: string | null;
  }
}
