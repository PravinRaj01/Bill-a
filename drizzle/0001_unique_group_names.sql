-- Lookalike groups (same name for the same user, differing only in case/spacing) must go
-- before the unique constraint below can be added. They came from re-saving a group that
-- had just been saved. Keep the NEWEST of each set: it holds the latest member list.
DELETE FROM "saved_groups" a USING "saved_groups" b
WHERE a.user_id = b.user_id
  AND lower(btrim(a.group_name)) = lower(btrim(b.group_name))
  AND (a.created_at < b.created_at OR (a.created_at = b.created_at AND a.id < b.id));--> statement-breakpoint
ALTER TABLE "saved_groups" ADD COLUMN "group_key" text GENERATED ALWAYS AS (lower(btrim(group_name))) STORED;--> statement-breakpoint
ALTER TABLE "saved_groups" ADD CONSTRAINT "saved_groups_user_key_uniq" UNIQUE("user_id","group_key");
