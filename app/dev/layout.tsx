import { notFound } from "next/navigation";

// Developer tools (the model bake-off) live under /dev. They read the user's API keys
// from localStorage and burn free-tier quota, so they must not exist in production.
// Set ENABLE_DEV_PAGES=1 to opt in (e.g. a private preview deployment).
export default function DevLayout({ children }: { children: React.ReactNode }) {
  if (process.env.NODE_ENV === "production" && process.env.ENABLE_DEV_PAGES !== "1") notFound();
  return children;
}
