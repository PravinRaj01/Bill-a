import os
import json
import base64  
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# 1. Load Environment Variables
load_dotenv()

# 2. Initialize App
app = FastAPI()

# 3. Updated Security (CORS) 
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",       # Keep for local testing
        "https://billa-rho.vercel.app"  # YOUR NEW VERCEL URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 4. Setup AI Models
vision_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    api_version="v1",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

chat_model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# 5. Data Structures
class SplitRequest(BaseModel):
    receipt_data: str
    user_instruction: str
    people_list: list
    apply_tax: bool = True  # Toggle for tax inclusion

# 6. Endpoints

@app.get("/")
def home():
    return {"status": "Bill-a Brain is Active"}

@app.post("/scan")
async def scan_receipt(file: UploadFile = File(...)):
    try:
        content = await file.read()
        base64_image = base64.b64encode(content).decode("utf-8")
        
        # PROMPT UPDATED: To handle quantities and line totals
        prompt = """
        Extract all items from this receipt image. 
        For each line item, identify:
        1. Quantity (e.g., 2)
        2. Name (e.g., CRAVING SET)
        3. Unit Price (price for one)
        4. Total Line Price (Quantity * Unit Price)

        Also extract:
        - The Currency Symbol used (e.g., RM, $, SGD, etc.)
        - Total Service Tax (SST) / Service Charge
        - Total Amount of the entire bill

        Return strictly valid JSON in this format:
        {
          "items": [
            {"name": "ITEM NAME", "quantity": 1, "unit_price": 0.0, "total_price": 0.0}
          ],
          "currency": "SYMBOL",
          "tax": 0.0,
          "total": 0.0
        }
        """

        image_data_uri = f"data:{file.content_type};base64,{base64_image}"
        msg = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_uri}}
        ])
        
        response = vision_model.invoke([msg])
        clean_json = response.content.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
        
    except Exception as e:
        print(f"Scan Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/split")
async def split_bill(request: SplitRequest):
    try:
        # EXTRACT CURRENCY FROM RECEIPT DATA
        receipt_obj = json.loads(request.receipt_data)
        curr = receipt_obj.get("currency", "RM")

        # STRICT PROMPT: Forces equal distribution and exact math matching
        prompt = """
        ACT AS A PRECISION BILL CALCULATOR.
        
        INPUTS:
        - PEOPLE: {people}
        - RECEIPT JSON: {receipt}
        - USER INSTRUCTION: {instruction}
        - APPLY TAX: {tax}
        - CURRENCY: {currency}
        
        STRICT CALCULATION RULES:
        1. TARGET TOTAL: Identify the 'Net Total' from the receipt (e.g., RM49.40). This is the absolute final sum. All individual shares MUST add up to this exact value.
        2. NEUTRAL SPLIT: If instructions are empty or say "split equally", you MUST divide every single item quantity by the number of people. (e.g., 1 Butter Chicken for 2 people = 0.5 each). Do NOT assign whole items to different people unless explicitly instructed.
        3. PROPORTIONAL TAX: If APPLY TAX is true, calculate tax based on the individual's food subtotal percentage.
        4. ROUNDING: If the sum of individual totals is off by 0.01 or 0.02 due to rounding, adjust the first person's total so the final sum matches the Receipt Total exactly.
        
        OUTPUT FORMAT:
        1. "Math Log": A very brief step-by-step of the calculation.
        2. A JSON array at the VERY END for the table UI:
        [
          {{"name": "PersonName", "amount": 0.00, "items": "Item A x0.5, Item B x1"}}
        ]
        """.format(
            people=request.people_list,
            receipt=request.receipt_data,
            instruction=request.user_instruction,
            tax=request.apply_tax,
            currency=curr
        )

        response = chat_model.invoke(prompt)
        return {"result": response.content}
    except Exception as e:
        print(f"Split Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))