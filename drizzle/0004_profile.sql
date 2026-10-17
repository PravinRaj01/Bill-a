ALTER TABLE "user" ADD COLUMN "username" text;--> statement-breakpoint
ALTER TABLE "user" ADD COLUMN "avatar_seed" text;--> statement-breakpoint
ALTER TABLE "user" ADD COLUMN "tz" text DEFAULT 'Asia/Kuala_Lumpur' NOT NULL;--> statement-breakpoint
ALTER TABLE "user" ADD CONSTRAINT "user_username_unique" UNIQUE("username");--> statement-breakpoint
ALTER TABLE "user" ADD CONSTRAINT "user_username_format" CHECK ("user"."username" ~ '^[a-z0-9_]{3,20}$');