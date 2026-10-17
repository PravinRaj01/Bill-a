import {
  boolean,
  check,
  index,
  integer,
  jsonb,
  pgTable,
  primaryKey,
  text,
  timestamp,
  unique,
  uniqueIndex,
  uuid,
} from "drizzle-orm/pg-core";
import { sql } from "drizzle-orm";
import type { AdapterAccountType } from "next-auth/adapters";
import type { BillData, PaymentDetails } from "@/types/domain";

// Single source of truth for the database. Two groups of tables:
//
// 1. Auth.js tables (users, accounts) — only these two, because we use JWT
//    sessions. @auth/drizzle-adapter requires just usersTable + accountsTable;
//    sessions / verificationTokens / authenticators are optional and only used
//    by database sessions. Skipping them means no session lookup on the auth
//    path, which matters because Neon's free tier scales to zero after 5
//    minutes and that can't be turned off.
//
// 2. App tables (saved_groups, bill_history). EVERY query against these must
//    filter on user_id from the verified server-side session — that rule is
//    what replaces Supabase RLS (plan §3.1). The schema can't enforce it, so
//    it lives in lib/actions/*.

// --- Auth.js ---------------------------------------------------------------

export const users = pgTable("user", {
  id: text("id")
    .primaryKey()
    .$defaultFn(() => crypto.randomUUID()),
  name: text("name"),
  email: text("email").unique(),
  emailVerified: timestamp("emailVerified", { mode: "date" }),
  image: text("image"),
  // Null for Google-only users. Argon2id hash for email+password users.
  passwordHash: text("password_hash"),
  // How friends can pay this user back (DuitNow QR/ID, bank, note). Validated by lib/validation/payment.ts
  // before it is written; null until the user sets something.
  paymentDetails: jsonb("payment_details").$type<PaymentDetails>(),
  // How friends find this person: lower-case a-z 0-9 _, 3 to 20 characters, unique. Null until claimed.
  username: text("username").unique(),
  // What the portrait is generated from (lib/ui/avatar.ts); null = generated from the name.
  avatarSeed: text("avatar_seed"),
  // The person's IANA time zone, so "today" and the daily digest use their own day.
  tz: text("tz").notNull().default("Asia/Kuala_Lumpur"),
}, (t) => [check("user_username_format", sql`${t.username} ~ '^[a-z0-9_]{3,20}$'`)]);

export const accounts = pgTable(
  "account",
  {
    userId: text("userId")
      .notNull()
      .references(() => users.id, { onDelete: "cascade" }),
    type: text("type").$type<AdapterAccountType>().notNull(),
    provider: text("provider").notNull(),
    providerAccountId: text("providerAccountId").notNull(),
    // Column names below are fixed by the Auth.js adapter contract.
    refresh_token: text("refresh_token"),
    access_token: text("access_token"),
    expires_at: integer("expires_at"),
    token_type: text("token_type"),
    scope: text("scope"),
    id_token: text("id_token"),
    session_state: text("session_state"),
  },
  (a) => [primaryKey({ columns: [a.provider, a.providerAccountId] })],
);

// --- App tables ------------------------------------------------------------

export const savedGroups = pgTable(
  "saved_groups",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    userId: text("user_id")
      .notNull()
      .references(() => users.id, { onDelete: "cascade" }),
    groupName: text("group_name").notNull(),
    // Case- and space-insensitive identity of the name ("Best Couple" == " best couple "),
    // computed by Postgres so it can never drift from group_name. A user can't have two
    // groups with the same key: saving an existing name updates that group instead of
    // creating a lookalike (which is what happened when Start Scanning was pressed twice).
    groupKey: text("group_key").generatedAlwaysAs(sql`lower(btrim(group_name))`),
    // The people OTHER than the user (Me is never stored in a name list).
    names: jsonb("names").$type<string[]>().notNull(),
    // Whether the user is part of this group: starting a split from it pre-adds the Me chip when true.
    includesMe: boolean("includes_me").notNull().default(true),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  },
  (t) => [
    index("saved_groups_user_created_idx").on(t.userId, t.createdAt.desc()),
    unique("saved_groups_user_key_uniq").on(t.userId, t.groupKey),
  ],
);

// People you split with: a name, an optional WhatsApp number, and (later) a link to their Bill.a account. A
// person is identified by their name ignoring case and spacing, like everywhere else in the app, so
// "RILEY " and "riley" are one contact. Scoped to the owner by user_id like every app table.
export const people = pgTable(
  "people",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    userId: text("user_id")
      .notNull()
      .references(() => users.id, { onDelete: "cascade" }),
    name: text("name").notNull(),
    nameKey: text("name_key").generatedAlwaysAs(sql`lower(btrim(name))`),
    // E.164 ("+60123456789"), validated in lib/people/phone.ts and again by this CHECK.
    phone: text("phone"),
    // Reserved for friends who have a Bill.a account (batch 5, friends PR).
    linkedUserId: text("linked_user_id").references(() => users.id, { onDelete: "set null" }),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
    updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
  },
  (t) => [
    unique("people_user_name_uniq").on(t.userId, t.nameKey),
    uniqueIndex("people_user_linked_uniq").on(t.userId, t.linkedUserId).where(sql`${t.linkedUserId} is not null`),
    index("people_user_idx").on(t.userId),
    check("people_phone_e164", sql`${t.phone} ~ '^\\+[1-9][0-9]{6,14}$'`),
    check("people_not_self", sql`${t.linkedUserId} <> ${t.userId}`),
  ],
);
export type Person = typeof people.$inferSelect;

export const billHistory = pgTable(
  "bill_history",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    userId: text("user_id")
      .notNull()
      .references(() => users.id, { onDelete: "cascade" }),
    // Generated on the client for the outbox. Unique PER USER (not globally):
    // a global unique would let one user's id collide with, or be probed
    // through, another's. pushOutbox upserts on (user_id, client_id).
    clientId: uuid("client_id").notNull(),
    billTitle: text("bill_title").notNull(),
    merchantCategory: text("merchant_category"),
    // Integer cents, matching the split engine. No floats anywhere.
    totalAmount: integer("total_amount").notNull(),
    currency: text("currency").notNull(),
    data: jsonb("data").$type<BillData>().notNull(),
    // Which tier produced the plan: "groq" | "gemini" | "fallback".
    providerTier: text("provider_tier"),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  },
  (t) => [
    unique("bill_history_user_client_uniq").on(t.userId, t.clientId),
    index("bill_history_user_created_idx").on(t.userId, t.createdAt.desc()),
  ],
);

export type User = typeof users.$inferSelect;
export type SavedGroup = typeof savedGroups.$inferSelect;
export type BillHistoryRow = typeof billHistory.$inferSelect;
