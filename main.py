import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# 1. Load Environment Variables (for local testing)
load_dotenv()

# 2. Initialize App
app = FastAPI()

# 3. Security (CORS) - Allows your future PWA to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, you will change this to your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Setup AI Models
# Vision: Gemini 2.5 Flash (Best free vision model)
# Note: If "gemini-2.5-flash" errors, switch to "gemini-1.5-flash"
vision_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp", 
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Logic: Llama 3.3 (Fastest free chat model)
chat_model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# 5. Data Structure
class SplitRequest(BaseModel):
    receipt_data: str
    user_instruction: str

# 6. Endpoints

@app.get("/")
def home():
    return {"status": "Bill Splitter Brain is Active"}

@app.post("/scan")
async def scan_receipt(file: UploadFile = File(...)):
    """
    Takes a photo, sends it to Gemini, returns a clean JSON list of items.
    """
    try:
        content = await file.read()
        
        # Strict prompt to force JSON output
        prompt = """
        Analyze this receipt image. Extract all items, prices, tax, and service charges.
        Return ONLY a raw JSON object with this exact structure:
        {
            "items": [{"name": "item_name", "price": 10.0, "quantity": 1}],
            "subtotal": 100.0,
            "tax": 10.0,
            "service_charge": 5.0,
            "total": 115.0,
            "currency": "$"
        }
        Do not include markdown formatting like ```json. Just return the raw JSON string.
        """

        msg = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": file.filename},
            {"type": "media", "mime_type": file.content_type, "data": content}
        ])
        
        response = vision_model.invoke([msg])
        
        # Clean the response just in case the model adds markdown
        clean_text = response.content.replace("```json", "").replace("```", "").strip()
        
        return json.loads(clean_text)

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/split")
async def calculate_split(req: SplitRequest):
    """
    Takes the receipt JSON + "Lim pays for drinks" and calculates the math.
    """
    try:
        system_prompt = """
        You are a bill splitting assistant. 
        I will provide the receipt data (JSON) and a User Instruction.
        
        RULES:
        1. Identify who pays for what based on the instruction.
        2. "Shared" items are split equally among all participants mentioned.
        3. Tax and Service Charge must be calculated proportionally based on each person's subtotal.
        4. Return a clean, formatted text summary suitable for copying to WhatsApp.
        """

        user_message = f"""
        Receipt Data: {req.receipt_data}
        User Instruction: {req.user_instruction}
        """

        response = chat_model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ])

        return {"result": response.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))