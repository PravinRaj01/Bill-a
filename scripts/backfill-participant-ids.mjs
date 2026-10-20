// One-off: records the account behind each existing share in the shared bill's participants, so shares follow the
// account instead of a name (see lib/db/social.ts syncSharesById). Safe to run again; only fills links that are missing.
//
//   node --env-file=.env.test.local scripts/backfill-participant-ids.mjs           (dry run: prints what it would change)
//   node --env-file=.env.test.local scripts/backfill-participant-ids.mjs --apply   (writes)
//
// Run it on the test branch first. On production, run it BEFORE deploying the code that reads the links, because a
// bill that has participants but no links would otherwise lose its share the next time its owner saves it.
import { neon } from "@neondatabase/serverless";

const url = process.env.DATABASE_URL;
if (!url) throw new Error("DATABASE_URL is not set (use --env-file)");
const apply = process.argv.includes("--apply");
const sql = neon(url);

const norm = (s) => String(s ?? "").trim().toLowerCase();

const rows = await sql.query(
  `select s.bill_id, s.user_id, s.as_name, b.data
     from bill_shares s join bill_history b on b.id = s.bill_id
    where b.data ? 'participants'`,
);

const byBill = new Map();
for (const r of rows) {
  const cur = byBill.get(r.bill_id) ?? { data: r.data, shares: [] };
  cur.shares.push({ userId: r.user_id, asName: r.as_name });
  byBill.set(r.bill_id, cur);
}

let changed = 0;
for (const [billId, { data, shares }] of byBill) {
  let touched = false;
  const participants = data.participants.map((p) => {
    if (p.me || p.userId) return p;
    const share = shares.find((s) => norm(s.asName) === norm(p.name));
    if (!share) return p;
    touched = true;
    return { ...p, userId: share.userId };
  });
  if (!touched) continue;
  changed++;
  console.log(`${apply ? "updating" : "would update"} bill ${billId}: ${participants.filter((p) => p.userId).map((p) => p.name).join(", ")}`);
  if (apply) await sql.query("update bill_history set data = $1::jsonb where id = $2", [JSON.stringify({ ...data, participants }), billId]);
}
console.log(`${changed} of ${byBill.size} shared bills with participants ${apply ? "updated" : "need updating (dry run, pass --apply to write)"}`);
