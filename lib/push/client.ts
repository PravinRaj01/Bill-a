import { removePushSubscription } from "@/lib/actions/push";
import { clearCachedPeople } from "@/lib/people/cache";

/**
 * Stops push on THIS device: removes its subscription from the signed-in account and from the browser. Used by the
 * notifications switch and before signing out, so a shared phone never keeps getting the previous user's pushes.
 * Never throws and never waits long: signing out must not hang on it.
 */
export async function dropThisDevicePush(): Promise<void> {
  await clearCachedPeople(); // the pals list is this account's: not left behind for the next person
  try {
    const reg = await navigator.serviceWorker?.getRegistration();
    const sub = await reg?.pushManager.getSubscription();
    if (!sub) return;
    await Promise.race([removePushSubscription(sub.endpoint).catch(() => false), new Promise((r) => setTimeout(r, 3000))]);
    await sub.unsubscribe().catch(() => false);
  } catch {
    /* nothing to drop, or the browser has no push */
  }
}
