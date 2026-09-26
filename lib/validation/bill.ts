import { z } from "zod";

// Shared by the save/sync server actions (a 'use server' file may only export
// async functions, so the schema lives here) and by the client outbox types.
// Cents are integers everywhere from here down. `data` is bounded so a client
// can't stash an arbitrarily large blob in our database.
export const billSchema = z.object({
  clientId: z.uuid(),
  billTitle: z.string().trim().min(1).max(120),
  merchantCategory: z.string().trim().max(40).nullish(),
  totalAmount: z.number().int().min(-100_000_000).max(100_000_000),
  currency: z.string().trim().min(1).max(8),
  providerTier: z.enum(["groq", "gemini", "fallback"]).nullish(),
  data: z.object({
    split: z
      .array(z.object({ name: z.string().max(60), amount: z.number().int(), items: z.string().max(2000) }))
      .max(50),
    items: z.object({
      items: z
        .array(
          z.object({
            name: z.string().max(200),
            quantity: z.number(),
            unitPrice: z.number().int(),
            totalPrice: z.number().int(),
          }),
        )
        .max(300),
      tax: z.number().int(),
      total: z.number().int(),
      currency: z.string().max(8),
    }),
    people: z.array(z.string().max(60)).max(50),
    reasoning: z.string().max(20_000),
  }),
});

export type BillPayload = z.input<typeof billSchema>;
