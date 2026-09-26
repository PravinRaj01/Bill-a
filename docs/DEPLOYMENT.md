# Deploying Bill.a to Vercel

Everything below the "What you do" line needs *your* accounts; nothing here needs a key
pasted into chat. Put secrets only into the Vercel dashboard / your local `.env.local`.

## Architecture in one paragraph

Next.js on Vercel (Singapore, `sin1`, next to Neon) · Neon Postgres over HTTP (Drizzle) ·
Auth.js v5 with JWT sessions (Google + email/password) · all AI is **bring-your-own-key**:
the browser calls Groq/Gemini directly with the user's key, so the server holds **no AI
keys** · OCR runs in the browser (tesseract.js WASM, files self-hosted under `/ocr/`) ·
a service worker caches the app shell and OCR runtime for offline scanning.

## 1. Database (Neon)

1. Neon console → your project → **Branches → Create branch** named `production`
   (keeps real users apart from the dev branch that tests create rows in).
2. Copy that branch's two connection strings (Connect → pooled and direct).
3. Apply the schema to it, from your machine:
   ```powershell
   $env:DATABASE_URL_UNPOOLED = "<production DIRECT string>"
   npx drizzle-kit migrate
   ```
   (`drizzle.config.ts` reads `DATABASE_URL_UNPOOLED`; the migration is `drizzle/0000_init.sql`.)
4. **Rotate the dev password** if it was ever pasted anywhere: Neon → Roles → `neondb_owner` → Reset password,
   then update both lines in `.env.local`.

## 2. Google sign-in

Google Cloud Console → APIs & Services → Credentials → your OAuth client → **Authorized redirect URIs**, add:

```
https://<your-production-domain>/api/auth/callback/google
```

(keep the `http://localhost:3000/...` one for local dev). Google does not allow wildcard redirect URIs,
so **Vercel preview URLs cannot use Google sign-in** — email/password works on previews.
If the consent screen is still in *Testing*, add testers, or publish it, before real users try.

## 3. Vercel project

1. Vercel → **Add New → Project** → import `PravinRaj01/Bill-a`, branch **`main`** (merge `monorepo` first) —
   or deploy the `monorepo` branch as a preview while you test.
2. Framework preset: Next.js (auto). Build/Install commands: defaults (`npm run build` also runs
   `ocr:assets` and `sw:build` via `prebuild`).
3. **Environment Variables** (Production; add to Preview too if you want previews to work):

   | Name | Value |
   |---|---|
   | `DATABASE_URL` | production **pooled** string |
   | `AUTH_SECRET` | a fresh `openssl rand -base64 32` (do **not** reuse the dev one) |
   | `AUTH_GOOGLE_ID` | from Google Cloud |
   | `AUTH_GOOGLE_SECRET` | from Google Cloud |

   Not needed on Vercel: `DATABASE_URL_UNPOOLED` (migrations only), `AUTH_URL` / `AUTH_TRUST_HOST`
   (Vercel sets the host automatically), any AI keys (there are none server-side).
   Leave `ENABLE_DEV_PAGES` unset so `/dev/*` returns 404.
4. Deploy. `vercel.json` pins functions to `sin1`.

## 4. Smoke test the live URL

- [ ] `/` loads; sign up with email + password; land on the dashboard
- [ ] "Continue with Google" reaches accounts.google.com and comes back signed in
- [ ] New session → scan a receipt photo → items appear (first scan downloads ~7 MB once)
- [ ] Settings → add a Groq key → **Test** says "Key works ✓" → split with an instruction
- [ ] Turn on airplane mode, **reload** → the app opens; a scan still works
- [ ] History shows the saved session; Continue restores it
- [ ] `https://<domain>/dev/model-bakeoff` is a 404
- [ ] Vercel → Logs: `{"telemetry":...}` lines appear and contain no names, prices or keys
- [ ] Browser DevTools → Network during a split: only `api.groq.com` / `generativelanguage.googleapis.com`
      requests carry the key; none to your own domain

## 5. After go-live

- Re-run the browser test suite against the live URL.
- Re-run `npm run live:check -- --bakeoff` whenever `lib/ai/providers/models.ts` changes.
- Known follow-ups: rename `middleware.ts` → `proxy.ts` (Next 16 deprecation warning, still works);
  email verification / password reset / login rate-limiting are intentionally out of v1.
