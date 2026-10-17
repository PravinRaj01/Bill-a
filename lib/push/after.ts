import { after } from "next/server";

/** Runs slower work once the response has gone out. Outside a request (tests, scripts) it just runs. */
export function afterResponse(task: () => Promise<unknown>): void {
  const run = () => task().catch(() => {});
  try {
    after(run);
  } catch {
    void run();
  }
}
