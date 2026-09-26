import {
  index,
  integer,
  jsonb,
  pgTable,
  primaryKey,
  text,
  timestamp,
  unique,
  uuid,
} from "drizzle-orm/pg-core";
import type { AdapterAccountType } from "next-auth/adapters";
import type { BillData } from "@/types/domain";

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
});

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
    names: jsonb("names").$type<string[]>().notNull(),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  },
  (t) => [index("saved_groups_user_created_idx").on(t.userId, t.createdAt.desc())],
);

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
