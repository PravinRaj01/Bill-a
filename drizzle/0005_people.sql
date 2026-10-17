CREATE TABLE "people" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" text NOT NULL,
	"name" text NOT NULL,
	"name_key" text GENERATED ALWAYS AS (lower(btrim(name))) STORED,
	"phone" text,
	"linked_user_id" text,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "people_user_name_uniq" UNIQUE("user_id","name_key"),
	CONSTRAINT "people_phone_e164" CHECK ("people"."phone" ~ '^\+[1-9][0-9]{6,14}$'),
	CONSTRAINT "people_not_self" CHECK ("people"."linked_user_id" <> "people"."user_id")
);
--> statement-breakpoint
ALTER TABLE "people" ADD CONSTRAINT "people_user_id_user_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."user"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "people" ADD CONSTRAINT "people_linked_user_id_user_id_fk" FOREIGN KEY ("linked_user_id") REFERENCES "public"."user"("id") ON DELETE set null ON UPDATE no action;--> statement-breakpoint
CREATE UNIQUE INDEX "people_user_linked_uniq" ON "people" USING btree ("user_id","linked_user_id") WHERE "people"."linked_user_id" is not null;--> statement-breakpoint
CREATE INDEX "people_user_idx" ON "people" USING btree ("user_id");