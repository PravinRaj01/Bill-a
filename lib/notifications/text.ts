import { formatMoney } from "@/lib/money";
import type { NotificationKind } from "@/lib/db/schema";

// The words and the destination for each kind of notification. Built here, from the kind and a small payload
// the server wrote, so what a friend sees is never text another user typed (apart from a bill's title).

export interface NotificationLike {
  kind: NotificationKind;
  billId: string | null;
  payload: Record<string, string | number>;
  /** Who did it (a friend of the reader), for kinds that open something about them. */
  actorId?: string | null;
}

const money = (p: Record<string, string | number>) => (typeof p.amount === "number" ? formatMoney(p.amount, String(p.currency || "RM")) : "");
const title = (p: Record<string, string | number>) => (typeof p.title === "string" && p.title ? `"${p.title}"` : "a bill");

/** The sentence, with the actor's name first ("Riley" or "@riley"). */
export function notificationText(n: NotificationLike, actor: string): string {
  switch (n.kind) {
    case "friend_request":
      return `${actor} sent you a friend request`;
    case "friend_accepted":
      return `${actor} accepted your friend request`;
    case "bill_added":
      return `${actor} added you to ${title(n.payload)}`;
    case "claim_made":
      return `${actor} says they paid you ${money(n.payload)} for ${title(n.payload)}`;
    case "claim_confirmed":
      return `${actor} confirmed your ${money(n.payload)} payment for ${title(n.payload)}`;
    case "claim_rejected":
      return `${actor} says your ${money(n.payload)} for ${title(n.payload)} hasn't arrived`;
    case "claim_changed":
      return `${actor} changed ${title(n.payload)}, so your payment note no longer matches. Check what you owe and tell them again`;
    case "bill_unshared":
      return `${actor} stopped sharing ${title(n.payload)} with you`;
    case "bill_deleted":
      return `${actor} deleted ${title(n.payload)}`;
    case "digest":
      return typeof n.payload.summary === "string" && n.payload.summary ? n.payload.summary : "Your daily summary";
    case "nudge":
      return `${actor} is reminding you about ${money(n.payload)} on ${title(n.payload)}`;
  }
}

/** Where tapping it goes, or null when there is nothing left to open. */
export function notificationHref(n: NotificationLike): string | null {
  switch (n.kind) {
    case "friend_request":
      return "/dashboard/people";
    case "friend_accepted":
      // Opens the offer to share past bills with the new friend.
      return n.actorId ? `/dashboard/people?share=${encodeURIComponent(n.actorId)}` : "/dashboard/people";
    case "claim_made":
      return n.billId ? `/dashboard/history/${n.billId}` : null;
    case "bill_added":
    case "claim_confirmed":
    case "claim_rejected":
    case "claim_changed":
    case "nudge":
      return n.billId ? `/dashboard/shared/${n.billId}` : null;
    case "digest":
      return "/dashboard/balances";
    case "bill_deleted":
    case "bill_unshared":
      return null;
  }
}
