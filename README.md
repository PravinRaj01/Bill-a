# Bill.a

AI-assisted bill splitting. Scan a receipt, say who owes what in plain English, get an
exact-to-the-cent split. Runs as a Next.js app with **no AI server of its own**: OCR runs
in your browser, and if you want AI-assisted splitting you paste in your own free Groq
or Gemini key, which the browser calls directly.

🔗 Live app: https://bill-a.vercel.app/

## What it does

- **Scan a receipt, entirely on your device.** [tesseract.js](https://github.com/naptha/tesseract.js)
  (WebAssembly) reads the photo locally — nothing is uploaded. A deterministic parser turns
  the OCR text into items, tax and a total, in integer cents. Works offline once the app has
  been opened online once.
- **Split it in plain English.** "Pravin pays for the drinks, split the rest equally" or
  "Wifey pays the tax." A deterministic reference resolver (`lib/retrieval/`) works out what
  "the drinks" or "the rest" refers to; the model only ever decides *who owns what* — never
  the arithmetic. All money math happens in one place, [`lib/split/engine.ts`](lib/split/engine.ts),
  using exact integer-cent fractions rounded once at the end, so a bill always reconciles to
  the printed total and never drifts a cent the way naive per-item rounding does.
- **Bring your own AI key, or don't.** Add a free [Groq](https://console.groq.com/keys) and/or
  [Google Gemini](https://aistudio.google.com/apikey) key in-app; it's stored only in your
  browser's `localStorage` and sent only to that provider, never to Bill.a's servers, which
  hold no AI key at all. Without a key, a deterministic on-device parser still handles clear
  instructions and asks a question instead of guessing on ambiguous ones.
- **Catches a confidently-wrong AI answer.** When the AI's plan and the on-device parser's
  plan disagree on the actual amounts, both readings are shown side by side and you pick —
  instead of silently trusting whichever answered first.
- **Cloud Enhance, opt-in.** If a scan comes out shaky, you can ask Gemini (your key) to
  re-read the photo. The photo is only sent when you tap that button.
- **Accounts, history, groups.** Google or email/password sign-in, saved settlement history,
  and named groups you can reuse — all scoped to your account server-side.

## Architecture

```
Browser
  ├─ tesseract.js (WASM)         receipt OCR, local, offline-capable
  ├─ deterministic parser        item/tax/total extraction + "the rest"/"the drinks" resolution
  ├─ split engine                exact-cent arithmetic — the only place money is computed
  ├─ Groq / Gemini (your key) ───────────────► called directly from the browser (CORS)
  └─ service worker               caches the app shell + OCR runtime for offline scanning

Next.js server (Vercel)
  ├─ Auth.js v5 (JWT sessions)    Google + email/password (argon2id)
  ├─ server actions               every query scoped to session.user.id — no AI keys touch this side
  └─ Neon Postgres (Drizzle)      history, groups, accounts
```

The AI cascade tries Groq first (fast, generous free tier), then Gemini, then falls back to
the on-device parser — never blocking a split on a slow or rate-limited provider. See
[`lib/ai/planSplit.ts`](lib/ai/planSplit.ts) for the full pipeline and
[`lib/ai/providers/models.ts`](lib/ai/providers/models.ts) for which models are current
(providers retire models every few months; that file is the only place the IDs live).

## Tech stack

- **Frontend**: Next.js 16 (App Router), TypeScript, React 19, Tailwind CSS v4, shadcn/ui
- **Auth & data**: Auth.js v5 (JWT), Drizzle ORM, Neon Postgres (serverless HTTP driver)
- **AI**: Groq (`openai/gpt-oss-20b`) and Google Gemini (`gemini-3.5-flash-lite`) — bring your
  own key, called directly from the browser; zod-validated on every response
- **OCR**: tesseract.js, self-hosted (not CDN-loaded), running in a Web Worker
- **Offline**: a hand-written service worker (`scripts/sw.template.js`) caches the OCR
  runtime and the guest-usable app shell; IndexedDB outbox for local-first bill saves
- **Deployment**: Vercel, single deploy, no separate backend service

## Project structure

```
Bill-a/
├── app/                    Next.js App Router — pages, layouts, server actions, API routes
│   ├── dashboard/          the app itself (new session, history, groups, account)
│   ├── api/auth/           Auth.js route handler
│   ├── api/telemetry/      privacy-preserving, allowlisted event logging
│   └── dev/                model bake-off tool — 404s in production
├── components/             UI components (shadcn/ui in components/ui/)
├── lib/
│   ├── ai/                 intent routing, prompts, schemas, validation, the provider cascade
│   ├── ai/providers/       Groq/Gemini callers, key store, Cloud Enhance
│   ├── split/               the arithmetic engine and the on-device fallback parser
│   ├── retrieval/           lexicon + reference resolution ("the drinks", "the rest")
│   ├── ocr/                 preprocessing, tesseract.js adapter, receipt-line parsing
│   ├── db/                  Drizzle schema, client, user-scoped queries
│   ├── actions/             server actions (the only way the client touches the database)
│   ├── sync/                IndexedDB outbox for local-first saves
│   └── telemetry/           the allowlisted event schema + client
├── bench/                   OCR and AI-model benchmarks (accuracy/latency), run against
│                            real receipts and real provider APIs — see bench/live-check.mts
├── drizzle/                 SQL migrations
├── scripts/                 build-time asset/service-worker generation
├── docs/DEPLOYMENT.md       step-by-step Vercel/Neon/Google Cloud deployment guide
└── legacy/python-backend/   archived FastAPI service — not built or deployed
```

## Getting started

### Prerequisites
- Node.js 20+
- A [Neon](https://neon.tech) Postgres project
- A Google OAuth client (for Google sign-in) — optional, email/password works without it

### Local development

```bash
git clone https://github.com/PravinRaj01/Bill-a.git
cd Bill-a
npm install
cp .env.example .env.local   # fill in DATABASE_URL, AUTH_SECRET, AUTH_GOOGLE_ID/SECRET
npx drizzle-kit migrate      # applies drizzle/ against DATABASE_URL_UNPOOLED
npm run dev
```

No AI key is required to run the app — split instructions fall back to the deterministic
parser. To test the AI cascade, add a Groq and/or Gemini key from inside the app (Settings),
or put `GROQ_API_KEY` / `GEMINI_API_KEY` in `.env.local` and run `npm run live:check` to
exercise the real provider APIs headlessly.

### Tests

```bash
npm test          # 448 tests: engine, parser, providers (mocked), auth-scoping, OCR, telemetry
npm run live:check              # real Groq/Gemini calls, using keys from .env.local
npm run live:check -- --bakeoff # scores each model on the accuracy bake-off cases
npm run bench:ocr               # OCR engine accuracy/latency on bench/receipts/
```

Cross-user data isolation and the split engine's cent-exact reconciliation are both covered
by dedicated test suites (`lib/db/queries.test.ts`, `lib/split/engine.exact.test.ts`) — the
former runs against a real Neon database when `DATABASE_URL` is set.

### Deploying

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for the full Vercel + Neon + Google OAuth
walkthrough, including the production smoke-test checklist.

## Security & privacy notes

- **No AI key ever reaches Bill.a's servers.** Keys live in `localStorage` and are sent only
  to `api.groq.com` / `generativelanguage.googleapis.com`, enforced by a strict
  `Content-Security-Policy` (see `next.config.ts`).
- **Every database query is scoped to the signed-in user**, server-side — there's no client-
  trusted user id anywhere in the query path.
- **Telemetry is an explicit allowlist** (`lib/telemetry/events.ts`): categorical facts only
  (which AI tier answered, an OCR confidence bucket), never receipt contents, names, amounts,
  or key material. Free-text error messages are redacted before logging.

## License

No license file yet — all rights reserved by default until one is added.

---
Built by PravinRaj
