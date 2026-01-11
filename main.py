import os
import json
import base64  
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
    model="gemini-1.5-flash", 
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
    try:
        content = await file.read()
        
        # Convert image bytes to base64 string
        base64_image = base64.b64encode(content).decode("utf-8")
        
        prompt = """
        Extract all items, prices, tax, and service charges from this receipt image.
        Return strictly valid JSON with this format:
        {"items": [{"name": "item", "price": 0.0, "quantity": 1}], "tax": 0.0, "total": 0.0}
        """

        # Gemini requires a specific data URI format for base64
        image_data_uri = f"data:{file.content_type};base64,{base64_image}"

        msg = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url", 
                "image_url": {"url": image_data_uri}  # This is the fix!
            }
        ])
        
        response = vision_model.invoke([msg])
        clean_json = response.content.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
        
    except Exception as e:
        print(f"Detailed Error: {e}") # This will show up in Koyeb logs
        raise HTTPException(status_code=500, detail=str(e))