# Bill.a

An AI-powered bill splitting and receipt management application built with Next.js, TypeScript, and Supabase. Bill.a simplifies group expenses by using OCR and LLMs to accurately split complex receipts, taxes, and service charges.

🔗 Live App: https://billa-rho.vercel.app/

## Features

### Core Functionality
- **AI Receipt Scanning**: Instant OCR extraction of items, quantities, and prices from receipt photos.
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
- **System Reasoning Log**: Transparent view of the AI's math and logic for every split.

## Tech Stack

- **Frontend**: Next.js 15 (App Router), TypeScript
- **Styling**: Tailwind CSS, shadcn/ui components
- **Database & Auth**: Supabase (PostgreSQL, Auth)
- **Backend API**: Python FastAPI (Deployed on Koyeb)
- **Icons**: Lucide React
- **Deployment**: Vercel

## Project Structure

src/
├── app/                # Next.js App Router (Pages, Layouts, Actions)
├── components/         # Reusable UI components
│   ├── ui/             # shadcn/ui primitives (Radix UI)
│   └── history/        # History list and selection logic
├── integrations/       # Supabase client and API definitions
├── utils/              # Helper functions and formatting
└── types/              # TypeScript type definitions

## Getting Started

### Prerequisites
- Node.js 18+
- npm / pnpm / bun

### Local Development

1. Clone the repository:
   git clone https://github.com/PravinRaj01/bill-a-web.git
   cd bill-a-web

2. Install dependencies:
   npm install

3. Set up environment variables (.env.local):
   NEXT_PUBLIC_SUPABASE_URL=your_project_url
   NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key

4. Run the development server:
   npm run dev

## License

This project is licensed under the MIT License.

---
Built with ❤️ by PravinRaj
