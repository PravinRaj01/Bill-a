import { signOutAction } from "@/app/auth/actions";
import { clearCachedPeople } from "@/lib/people/cache";
import { dropThisDevicePush } from "@/lib/push/client";

/**
 * Signs out and leaves nothing of this account on the device: the service worker's cached page shells are wiped, the
 * saved pals list is removed, and this phone stops receiving the account's pushes. Every sign-out button uses this,
 * so a shared phone behaves the same whichever one was tapped. The clean-up never throws and never waits long, so
 * signing out cannot hang on it.
 */
export async function signOutCleanly(): Promise<void> {
  try {
    navigator.serviceWorker?.controller?.postMessage({ type: "clear-pages" });
  } catch {
    /* no service worker: nothing cached */
  }
  await clearCachedPeople();
  await dropThisDevicePush();
  await signOutAction();
}
