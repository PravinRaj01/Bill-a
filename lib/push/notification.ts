import { eq } from "drizzle-orm";

import type { Db } from "@/lib/db/client";
import { users } from "@/lib/db/schema";
import type { NotifyInput } from "@/lib/db/notifications";
import { notificationHref, notificationText } from "@/lib/notifications/text";
import { ownerLabel } from "@/lib/split/shared";
import { pushConfigured, pushToUser } from "./send";

/** Pushes a freshly recorded inbox notification to the recipient's devices, in the same words as the inbox. */
export async function pushNotification(db: Db, n: NotifyInput): Promise<void> {
  if (!pushConfigured()) return;
  try {
    let actor = "Someone";
    if (n.actorId) {
      const [a] = await db.select({ name: users.name, username: users.username }).from(users).where(eq(users.id, n.actorId));
      if (a) actor = ownerLabel(a);
    }
    const like = { kind: n.kind, billId: n.billId ?? null, payload: n.payload ?? {} };
    await pushToUser(db, n.userId, { title: "Bill.a", body: notificationText(like, actor), url: notificationHref(like) ?? "/dashboard/notifications", tag: n.dedupe });
  } catch {
    /* the inbox row is already saved; the push is a bonus */
  }
}
