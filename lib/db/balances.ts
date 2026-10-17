import type { Db } from "./client";
import { listTrackedBills } from "./queries";
import { listSharedBills } from "./social";
import { aggregateBalances, type PersonBalance } from "@/lib/split/settle";
import { sharedBalances } from "@/lib/split/shared";

/** Net balances for one user across their own tracked bills and friends' shared bills, biggest first. */
export async function balancesFor(db: Db, userId: string, today: string): Promise<{ people: PersonBalance[]; own: number; shared: number; withoutMe: number }> {
  const [rows, shared] = await Promise.all([listTrackedBills(db, userId), listSharedBills(db, userId)]);
  const own = aggregateBalances(
    rows.map((r) => ({ id: r.id, title: r.billTitle, createdAt: r.createdAt, currency: r.currency, split: r.split ?? [], settlement: r.settlement ?? undefined })),
    today,
  );
  // Friends' bills shared with me count too, but only what runs between me and the friend who owns each one.
  const fromFriends = sharedBalances(shared.map((b) => ({ id: b.id, title: b.billTitle, createdAt: b.createdAt, currency: b.currency, split: b.split, settlement: b.settlement, participants: b.participants, asName: b.asName, owner: b.owner })), today);
  const people = [...own, ...fromFriends].sort((a, b) => Math.abs(b.net) - Math.abs(a.net) || a.name.localeCompare(b.name));
  return { people, own: rows.length, shared: shared.filter((b) => b.settlement).length, withoutMe: rows.filter((r) => !r.settlement?.me && !r.participants).length };
}
