import { auth } from "@/lib/auth";

export class UnauthorizedError extends Error {
  constructor() {
    super("Unauthorized");
    this.name = "UnauthorizedError";
  }
}

/**
 * The ONLY source of a userId for database access. It comes from the signed,
 * server-verified session — never from a parameter the client sent. Server
 * actions are public HTTP endpoints, so anything they accept as an argument is
 * attacker-controlled; the user's identity must not be one of those arguments.
 */
export async function requireUserId(): Promise<string> {
  const id = (await auth())?.user?.id;
  if (!id) throw new UnauthorizedError();
  return id;
}

/** For read actions that guests may call: null instead of throwing. */
export async function getUserIdOrNull(): Promise<string | null> {
  return (await auth())?.user?.id ?? null;
}
