import { openDB, type IDBPDatabase } from "idb";

// The pals list, kept on the device so the people picker works offline. It is a copy of what the server last said
// (refreshed whenever the app is online) and is removed when the user signs out, so nothing is left for the next
// person on a shared device. Never throws: with no IndexedDB (private window, old browser) there is just no cache.

const DB_NAME = "billa-people";
const STORE = "people";
const KEY = "list";

let dbPromise: Promise<IDBPDatabase> | null = null;
function db(): Promise<IDBPDatabase> {
  dbPromise ??= openDB(DB_NAME, 1, {
    upgrade(d) {
      d.createObjectStore(STORE);
    },
  });
  return dbPromise;
}

export async function readCachedPeople<T>(): Promise<T[] | null> {
  try {
    const rows = await (await db()).get(STORE, KEY);
    return Array.isArray(rows) ? (rows as T[]) : null;
  } catch {
    return null;
  }
}

export async function writeCachedPeople<T>(rows: T[]): Promise<void> {
  try {
    await (await db()).put(STORE, rows, KEY);
  } catch {
    /* no cache */
  }
}

export async function clearCachedPeople(): Promise<void> {
  try {
    await (await db()).delete(STORE, KEY);
  } catch {
    /* nothing to clear */
  }
}
