// Where to go after logging in. Auth.js sends a logged-out visitor to "/" with ?callbackUrl=<the page they wanted>,
// so an invite link (/dashboard/add/alex) survives the login. Only a path inside /dashboard is ever followed: the
// value is reduced to its path and query (any host is dropped), so it can never send someone to another site.
export function safeCallback(raw: unknown, fallback = "/dashboard"): string {
  if (typeof raw !== "string" || raw.length === 0 || raw.length > 500 || raw.includes("\\")) return fallback;
  try {
    const u = new URL(raw, "http://localhost");
    const ok = u.pathname === "/dashboard" || u.pathname.startsWith("/dashboard/");
    return ok && !u.pathname.includes("//") ? u.pathname + u.search : fallback;
  } catch {
    return fallback;
  }
}
