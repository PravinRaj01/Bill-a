# Bill.a Backend API

The high-performance processing engine for Bill.a. This API handles OCR extraction and utilizes Large Language Models to interpret complex receipt structures and user splitting instructions.

## Core Services

### OCR & Extraction
- **Receipt Parsing**: Converts raw images into structured JSON data.
- **Item Detection**: Identifies names, unit prices, and total prices accurately.
- **Tax Extraction**: Automatically detects tax and service charge sub-lines.

### AI Splitting Logic
- **Natural Language Processing**: Interprets user instructions to handle complex edge cases.
- **Mathematical Validation**: Ensures the sum of split amounts matches the receipt total.
- **Reasoning Generation**: Provides a step-by-step log of the calculation for transparency.

## Tech Stack

- **Language**: Python 3.10+
- **Framework**: FastAPI
- **AI Integration**: OpenAI GPT-4o / Google Gemini API
- **OCR Engine**: Tesseract OCR / Google Vision
- **Deployment**: Koyeb (Serverless)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/scan`  | POST   | Upload image -> Returns structured JSON items |
| `/split` | POST   | Receipt Data + Instructions -> Returns Settlement |
| `/health`| GET    | Service health check |

## Local Setup

1. Clone the repository:
   git clone https://github.com/PravinRaj01/Bill-a.git
   cd Bill-a

2. Install requirements:
   pip install -r requirements.txt

3. Run with Uvicorn:
   uvicorn main:app --host 0.0.0.0 --port 8000

## Usage

The backend is designed to be called by the Bill.a frontend. It processes multipart/form-data for image scans and JSON payloads for splitting logic.

---
Built by PravinRaj
