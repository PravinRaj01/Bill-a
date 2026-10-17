CREATE TABLE "bill_shares" (
	"bill_id" uuid NOT NULL,
	"user_id" text NOT NULL,
	"as_name" text NOT NULL,
	"hidden" boolean DEFAULT false NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "bill_shares_bill_id_user_id_pk" PRIMARY KEY("bill_id","user_id")
);
--> statement-breakpoint
CREATE TABLE "friendships" (
	"user_a" text NOT NULL,
	"user_b" text NOT NULL,
	"requested_by" text NOT NULL,
	"status" text DEFAULT 'pending' NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"accepted_at" timestamp with time zone,
	CONSTRAINT "friendships_user_a_user_b_pk" PRIMARY KEY("user_a","user_b"),
	CONSTRAINT "friendships_ordered" CHECK ("friendships"."user_a" < "friendships"."user_b"),
	CONSTRAINT "friendships_requester" CHECK ("friendships"."requested_by" in ("friendships"."user_a", "friendships"."user_b")),
	CONSTRAINT "friendships_status" CHECK ("friendships"."status" in ('pending', 'accepted', 'declined'))
);
--> statement-breakpoint
CREATE TABLE "payment_claims" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"bill_id" uuid NOT NULL,
	"user_id" text NOT NULL,
	"transfer_key" text NOT NULL,
	"amount" integer NOT NULL,
	"status" text DEFAULT 'pending' NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"resolved_at" timestamp with time zone,
	CONSTRAINT "payment_claims_uniq" UNIQUE("bill_id","user_id","transfer_key"),
	CONSTRAINT "payment_claims_status" CHECK ("payment_claims"."status" in ('pending', 'confirmed', 'rejected'))
);
--> statement-breakpoint
ALTER TABLE "bill_history" ADD COLUMN "updated_at" timestamp with time zone DEFAULT now() NOT NULL;--> statement-breakpoint
ALTER TABLE "bill_shares" ADD CONSTRAINT "bill_shares_bill_id_bill_history_id_fk" FOREIGN KEY ("bill_id") REFERENCES "public"."bill_history"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "bill_shares" ADD CONSTRAINT "bill_shares_user_id_user_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."user"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "friendships" ADD CONSTRAINT "friendships_user_a_user_id_fk" FOREIGN KEY ("user_a") REFERENCES "public"."user"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "friendships" ADD CONSTRAINT "friendships_user_b_user_id_fk" FOREIGN KEY ("user_b") REFERENCES "public"."user"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "friendships" ADD CONSTRAINT "friendships_requested_by_user_id_fk" FOREIGN KEY ("requested_by") REFERENCES "public"."user"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "payment_claims" ADD CONSTRAINT "payment_claims_share_fk" FOREIGN KEY ("bill_id","user_id") REFERENCES "public"."bill_shares"("bill_id","user_id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
CREATE INDEX "bill_shares_user_idx" ON "bill_shares" USING btree ("user_id");--> statement-breakpoint
CREATE INDEX "friendships_user_b_idx" ON "friendships" USING btree ("user_b");--> statement-breakpoint
CREATE INDEX "payment_claims_bill_idx" ON "payment_claims" USING btree ("bill_id");