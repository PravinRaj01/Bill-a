# Bill.a

An AI-powered bill splitting and receipt management application built with Next.js, TypeScript, and Supabase.

🔗 Live App: https://billa-rho.vercel.app/

> **Status: mid-migration.** This repo was just consolidated from two separate repositories (Next.js frontend + Python FastAPI backend) into a single monorepo, and is being migrated to a 100% client-side AI architecture (WebLLM + Transformers.js, no backend server). See below for what's implemented today vs. what's in progress.

## Features

### Core Functionality
- **AI Receipt Scanning**: Extraction of items, quantities, and prices from receipt photos.
- **Smart Tax Handling**: Dynamic toggle for Tax & Service Charges with real-time grand total updates.
- **Natural Language Splitting**: Instruct the AI in plain English (e.g., "Pravin pays for the drinks, split the rest equally").
- **Guest Access**: Full calculator functionality available without requiring an account.
- **WhatsApp Integration**: One-tap sharing of formatted settlement summaries to group chats.

### User Features (Authenticated)
- **Session History**: Persistent storage of past split sessions with detailed breakdowns.
- **Saved Groups**: Save frequently used groups of friends to skip name entry.
- **Continue Session**: Re-open past sessions to add new receipts to the same group of people.
- **Session Naming**: Automatic and custom naming logic for organized history tracking (Session 1...n).

### Interactive Management
- **Bulk History Actions**: Selection mode to delete individual, multiple, or all history records.
- **System Reasoning Log**: Transparent view of the math and logic behind every split.

## Tech Stack

- **Frontend**: Next.js (App Router), TypeScript, React 19
- **Styling**: Tailwind CSS v4, shadcn/ui components
- **Database & Auth**: Supabase (PostgreSQL, Auth)
- **AI (in progress)**: WebLLM (WebGPU, in-browser LLM) for split logic, Transformers.js (ONNX, WASM) for retrieval embeddings, client-side OCR — replacing the legacy Python backend below
- **Icons**: Lucide React
- **Deployment**: Vercel (single deploy — no separate backend service)

## Project Structure

```
Bill-a/
├── app/                       Next.js App Router (pages, layouts, auth)
├── components/                Reusable UI components (shadcn/ui in components/ui/)
├── hooks/
├── lib/                       Client utilities, Supabase clients
├── utils/supabase/            Supabase browser/server client setup
├── legacy/python-backend/     Archived FastAPI service — no longer deployed
├── middleware.ts
└── public/
```

`legacy/python-backend/` is kept for reference (git history and prior prompt logic) but is not built or deployed. The app currently still calls its previously-hosted Koyeb API for scan/split during the migration; this is being replaced endpoint-by-endpoint with local, in-browser processing.

## Getting Started

### Prerequisites
- Node.js 18+
- npm

### Local Development

```bash
git clone https://github.com/PravinRaj01/Bill-a.git
cd Bill-a
npm install
```

Set up environment variables (`.env.local`):

```
NEXT_PUBLIC_SUPABASE_URL=your_project_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key
```

Run the development server:

```bash
npm run dev
```

## License

This project is licensed under the MIT License.

---
Built with ❤️ by PravinRaj
