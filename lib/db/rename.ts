import { and, eq, sql } from "drizzle-orm";

import { renamedForm, renameInBill, renameTransferKey } from "@/lib/people/rename";
import { samePerson } from "@/lib/split/participants";
import type { BillData } from "@/types/domain";
import type { Db } from "./client";
import { billHistory, billShares, paymentClaims, people, savedGroups } from "./schema";

// A user renamed themselves: their friends' records follow. Only where the friend wrote them by their old full
// or first name (lib/people/rename.ts); a nickname a friend chose stays. Touches only rows that point at this
// user: bills shared with them, contacts linked to them, and those owners' saved groups.

export interface RenameResult {
  bills: number;
  contacts: number;
  groups: number;
}

export async function propagateRename(db: Db, userId: string, oldName: string | null, newName: string | null): Promise<RenameResult> {
  const done: RenameResult = { bills: 0, contacts: 0, groups: 0 };
  if (!oldName?.trim() || !newName?.trim() || samePerson(oldName, newName)) return done;

  // Bills shared with them: rewrite every place the bill names them, the share, and their payment claims.
  const shared = await db
    .select({ billId: billShares.billId, asName: billShares.asName, data: billHistory.data })
    .from(billShares)
    .innerJoin(billHistory, eq(billHistory.id, billShares.billId))
    .where(eq(billShares.userId, userId));
  for (const s of shared) {
    const to = renamedForm(s.asName, oldName, newName);
    const data = to ? renameInBill(s.data as BillData, s.asName, to) : null;
    if (!to || !data) continue;
    const claims = await db.select({ id: paymentClaims.id, transferKey: paymentClaims.transferKey }).from(paymentClaims).where(and(eq(paymentClaims.billId, s.billId), eq(paymentClaims.userId, userId)));
    // One batch per bill: the bill, the share and the claims change together or not at all. updated_at moves too, so
    // the owner's other devices see a changed bill.
    const writes = [
      db.update(billHistory).set({ data, updatedAt: sql`now()` }).where(eq(billHistory.id, s.billId)),
      db.update(billShares).set({ asName: to }).where(and(eq(billShares.billId, s.billId), eq(billShares.userId, userId))),
      ...claims.flatMap((c) => {
        const key = renameTransferKey(c.transferKey, s.asName, to);
        return key !== c.transferKey ? [db.update(paymentClaims).set({ transferKey: key }).where(eq(paymentClaims.id, c.id))] : [];
      }),
    ];
    await db.batch(writes as unknown as [(typeof writes)[0], ...(typeof writes)[number][]]);
    done.bills++;
  }

  // Their contact in each friend's People, and that friend's saved groups that list them.
  const contacts = await db.select({ id: people.id, ownerId: people.userId, name: people.name }).from(people).where(eq(people.linkedUserId, userId));
  for (const c of contacts) {
    const to = renamedForm(c.name, oldName, newName);
    if (!to) continue;
    try {
      await db.update(people).set({ name: to }).where(eq(people.id, c.id));
      done.contacts++;
    } catch {
      continue; // the friend already has a contact with the new name: leave theirs alone
    }
    const groups = await db.select({ id: savedGroups.id, names: savedGroups.names }).from(savedGroups).where(eq(savedGroups.userId, c.ownerId));
    for (const g of groups) {
      if (!g.names.some((n) => samePerson(n, c.name)) || g.names.some((n) => samePerson(n, to))) continue;
      await db.update(savedGroups).set({ names: g.names.map((n) => (samePerson(n, c.name) ? to : n)) }).where(eq(savedGroups.id, g.id));
      done.groups++;
    }
  }
  return done;
}
