import { aggregateBalances, type PersonBalance } from "@/lib/split/settle";
import { sharedBalances } from "@/lib/split/shared";
import type { Db } from "./client";
import { listTrackedBills } from "./queries";
import { listSharedBills } from "./social";

export interface UserBalances {
  people: PersonBalance[];
  /** Bills counted: my own tracked ones plus friends' shared bills that are tracked. */
  tracked: number;
  /** Older tracked bills that don't say which person is you, so they can't count. */
  withoutMe: number;
}

/**
 * What each person owes this user (or the user them) across tracked bills, plus friends' shared bills (only what
 * runs between me and the friend who owns each). One place, used by Balances and by the daily digest.
 */
export async function computeBalances(db: Db, userId: string, today: string): Promise<UserBalances> {
  const [rows, shared] = await Promise.all([listTrackedBills(db, userId), listSharedBills(db, userId)]);
  const own = aggregateBalances(
    rows.map((r) => ({ id: r.id, title: r.billTitle, createdAt: r.createdAt, currency: r.currency, split: r.split ?? [], settlement: r.settlement ?? undefined })),
    today,
  );
  const fromFriends = sharedBalances(
    shared.map((b) => ({ id: b.id, title: b.billTitle, createdAt: b.createdAt, currency: b.currency, split: b.split, settlement: b.settlement, participants: b.participants, asName: b.asName, owner: b.owner })),
    today,
  );
  const people = [...own, ...fromFriends].sort((a, b) => Math.abs(b.net) - Math.abs(a.net) || a.name.localeCompare(b.name));
  return { people, tracked: rows.length + shared.filter((b) => b.settlement).length, withoutMe: rows.filter((r) => !r.settlement?.me && !r.participants).length };
}
