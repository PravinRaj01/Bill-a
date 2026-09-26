/* Bill.a service worker.
 *
 * Source of truth is scripts/sw.template.js; scripts/build-sw.mjs stamps the build id
 * and OCR version in and writes public/sw.js (gitignored) before every dev/build.
 *
 * What it does — and, deliberately, what it doesn't:
 *  - Caches static assets (Next's content-hashed /_next/static/*, icons, manifest) and
 *    the self-hosted OCR runtime (/ocr/*: worker, WASM core, language data) so that after
 *    ONE online visit the receipt scanner works with no signal at all.
 *  - Keeps a copy of the two guest-friendly app shells (/dashboard/new, /dashboard) so a
 *    cold start with no network still opens the app. Network-first: a fresh page always
 *    wins when the network is there.
 *  - NEVER touches: non-GET requests (server actions are POSTs), /api/*, Next's RSC/prefetch
 *    requests, or per-user pages (History, Account, History detail). Those must come from the
 *    server, every time, so one person's data can't be served to another on a shared phone.
 *  - The cached shells are wiped on sign-out (the page posts "clear-pages").
 */

const BUILD_ID = "__BUILD_ID__";
const OCR_VERSION = "__OCR_VERSION__";

const STATIC = "billa-static-v1"; //           content-hashed files: safe to keep across deploys
const OCR = `billa-ocr-${OCR_VERSION}`; //     ~7 MB: only re-downloaded when the OCR packages change
const PAGES = `billa-pages-${BUILD_ID}`; //    HTML shells: replaced every deploy

const SHELL_ROUTES = ["/dashboard/new", "/dashboard"];
const STATIC_FILES = new Set(["/icon.png", "/apple-icon.png", "/favicon.ico", "/manifest.json"]);
const MAX_STATIC_ENTRIES = 250; // old deploys' chunks would otherwise pile up forever
const SLOW_NETWORK_MS = 4000; //  with a cached shell in hand, don't make the user wait longer than this

self.addEventListener("install", () => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const keep = new Set([STATIC, OCR, PAGES]);
      for (const name of await caches.keys()) {
        if (name.startsWith("billa-") && !keep.has(name)) await caches.delete(name);
      }
      await self.clients.claim();
      await cachePage("/dashboard/new"); // best effort: the offline shell exists right after install
    })(),
  );
});

self.addEventListener("message", (event) => {
  const data = event.data || {};
  if (data.type === "clear-pages") event.waitUntil(caches.delete(PAGES));
  if (data.type === "cache-page" && SHELL_ROUTES.includes(data.path)) event.waitUntil(cachePage(data.path));
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;

  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;

  const path = url.pathname;
  if (path.startsWith("/api/")) return;
  if (url.searchParams.has("_rsc") || req.headers.has("rsc") || req.headers.has("next-router-prefetch")) return;

  if (path.startsWith("/ocr/")) {
    event.respondWith(cacheFirst(event, OCR, false));
  } else if (path.startsWith("/_next/static/") || STATIC_FILES.has(path)) {
    event.respondWith(cacheFirst(event, STATIC, path.startsWith("/_next/static/")));
  } else if (req.mode === "navigate") {
    event.respondWith(navigate(event, url));
  }
});

// ---------------------------------------------------------------------------

async function cacheFirst(event, cacheName, trim) {
  const cache = await caches.open(cacheName);
  const hit = await cache.match(event.request);
  if (hit) return hit;

  const res = await fetch(event.request);
  // Only whole, successful responses (a 206 partial or an error must never be cached).
  if (res.ok && res.status === 200) {
    event.waitUntil(
      (async () => {
        await cache.put(event.request, res.clone());
        if (trim) await trimCache(cache);
      })(),
    );
  }
  return res;
}

async function trimCache(cache) {
  const keys = await cache.keys();
  for (const key of keys.slice(0, Math.max(0, keys.length - MAX_STATIC_ENTRIES))) await cache.delete(key);
}

function isCacheableHtml(res) {
  return res.ok && !res.redirected && (res.headers.get("content-type") || "").includes("text/html");
}

async function cachePage(path) {
  try {
    const res = await fetch(path, { credentials: "same-origin", cache: "no-store" });
    if (!isCacheableHtml(res)) return;
    const html = await res.clone().text();
    await (await caches.open(PAGES)).put(path, res);
    // The page's own scripts/styles/fonts were fetched BEFORE this worker controlled the page,
    // so they never passed through cacheFirst(). Without them a cached shell would open to a
    // blank, script-less page offline. Pull in everything the HTML (and its CSS) references.
    await precacheStatic(html);
  } catch (_) {
    /* offline at install time: the next online visit will cache it */
  }
}

const STATIC_REF = /\/_next\/static\/[^"'\s\\)<>]+/g;

async function precacheStatic(text) {
  const cache = await caches.open(STATIC);
  const seen = new Set();
  const queue = [...new Set(text.match(STATIC_REF) || [])];
  while (queue.length && seen.size < MAX_STATIC_ENTRIES) {
    const ref = queue.shift();
    if (seen.has(ref)) continue;
    seen.add(ref);
    try {
      let res = await cache.match(ref);
      if (!res) {
        res = await fetch(ref);
        if (!(res.ok && res.status === 200)) continue;
        await cache.put(ref, res.clone());
      }
      // CSS references its fonts/images by url(): follow one more level.
      if (ref.split("?")[0].endsWith(".css")) {
        for (const nested of (await res.clone().text()).match(STATIC_REF) || []) queue.push(nested);
      }
    } catch (_) {
      /* skip anything we can't fetch */
    }
  }
}

async function navigate(event, url) {
  const isShell = SHELL_ROUTES.includes(url.pathname);
  const pages = await caches.open(PAGES);
  const cached = isShell ? await pages.match(url.pathname) : undefined;

  try {
    const network = fetch(event.request);
    const res = cached
      ? await Promise.race([
          network,
          new Promise((_, reject) => setTimeout(() => reject(new Error("slow")), SLOW_NETWORK_MS)),
        ])
      : await network;
    if (isShell && isCacheableHtml(res)) event.waitUntil(pages.put(url.pathname, res.clone()));
    return res;
  } catch (_) {
    if (cached) return cached;
    // The installed app opens "/": send it to the offline-capable shell if we have one.
    if (url.pathname === "/" && (await pages.match("/dashboard/new"))) {
      return Response.redirect(new URL("/dashboard/new", self.location.origin).href, 302);
    }
    return offlinePage();
  }
}

function offlinePage() {
  const html =
    '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' +
    "<title>Offline · Bill.a</title>" +
    '<body style="font-family:system-ui;background:#000;color:#fff;display:grid;place-items:center;min-height:100vh;margin:0;text-align:center;padding:24px">' +
    "<div><h1 style=\"font-size:18px\">You're offline</h1>" +
    '<p style="color:#71717a;font-size:14px">This page needs a connection. Splitting a new bill still works offline.</p>' +
    '<p><a href="/dashboard/new" style="color:#fff">Start a new session</a></p></div>';
  return new Response(html, { status: 503, headers: { "Content-Type": "text/html; charset=utf-8" } });
}
