import type { NextAuthConfig } from "next-auth";

// Edge-safe half of the Auth.js config. middleware.ts imports THIS, not
// lib/auth.ts — the full config pulls in the DB adapter and the native
// argon2 binding, neither of which can load on the edge runtime. With JWT
// sessions, middleware only ever needs to decode the signed cookie, so it
// needs no provider, no adapter and no database. That is what keeps Neon's
// scale-to-zero cold start off the auth path entirely.

export const authConfig = {
  // Unauthenticated users hitting a protected route land on the login page,
  // and Auth.js errors (e.g. OAuthAccountNotLinked) come back as /?error=...
  pages: { signIn: "/", error: "/" },
  session: { strategy: "jwt" },
  providers: [], // real providers are added in lib/auth.ts
  callbacks: {
    // Same policy the old Supabase middleware enforced: the dashboard and
    // new-split flow are open to guests, history/account require a login.
    authorized({ auth, request: { nextUrl } }) {
      const path = nextUrl.pathname;
      const loggedIn = !!auth?.user;

      const strictlyProtected = ["/dashboard/history", "/dashboard/account", "/dashboard/balances", "/dashboard/people"];
      if (strictlyProtected.some((p) => path.startsWith(p))) return loggedIn;

      if (loggedIn && path === "/") {
        return Response.redirect(new URL("/dashboard", nextUrl));
      }
      return true;
    },
    // Copy the user id into the token, then onto the session, so server code
    // can scope every query with session.user.id (plan §3.1).
    jwt({ token, user, trigger, session }) {
      if (user?.id) token.sub = user.id;
      // The signed-in user changed their own name (lib/actions/user.ts updateProfileName): the token is
      // made at sign-in, so without this the old name would stay in the session until the next login.
      if (trigger === "update" && session?.user && "name" in session.user) token.name = session.user.name ?? null;
      // Same for the profile (lib/actions/user.ts updateProfile): username and portrait.
      if (trigger === "update" && session?.user && "username" in session.user) token.username = session.user.username ?? null;
      if (trigger === "update" && session?.user && "avatarSeed" in session.user) token.avatarSeed = session.user.avatarSeed ?? null;
      return token;
    },
    session({ session, token }) {
      if (token.sub) session.user.id = token.sub;
      session.user.username = token.username ?? null;
      session.user.avatarSeed = token.avatarSeed ?? null;
      return session;
    },
  },
} satisfies NextAuthConfig;
