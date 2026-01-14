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
origins = [
    "http://localhost:3000",                  # For local testing
    "https://billa-rho.vercel.app",           # Your actual Vercel Production URL
    "https://billa-rho.vercel.app/",          # Just in case (trailing slash)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,                    # Allow these specific websites
    allow_credentials=True,
    allow_methods=["*"],                      # Allow all methods (POST, GET, OPTIONS)
    allow_headers=["*"],                      # Allow all headers
)
# 4. Setup AI Models
vision_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
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
        receipt_obj = json.loads(request.receipt_data)
        curr = receipt_obj.get("currency", "RM")

        prompt = """
        ACT AS A SENIOR AUDITOR AND PRECISION BILL CALCULATOR.
        
        GOAL: Calculate exactly how much each person owes based on the receipt and instructions.
        
        INPUTS:
        - PEOPLE: {people}
        - RECEIPT: {receipt}
        - SPECIAL INSTRUCTIONS: {instruction}
        - TAX/SERVICE CHARGE INCLUDED: {tax_enabled}
        
        CORE ALGORITHM:
        1. ASSIGNMENT: Read the 'SPECIAL INSTRUCTIONS'. If an item is assigned to a person, they get 100% of that quantity. For all items NOT mentioned in instructions, divide their quantity equally among ALL people (e.g., 1 item / 4 people = 0.25 each).
        
        2. FOOD SUBTOTAL: For each person, sum (Assigned Quantity * Unit Price). This is their 'Raw Food Cost'.
        
        3. TAX RATIO: 
           - Calculate the 'Total Raw Food Cost' of all items.
           - Calculate the 'Tax/Service Amount' from the receipt (Receipt Total - Total Raw Food Cost).
           - Tax Percentage = (Tax/Service Amount / Total Raw Food Cost).
        
        4. INDIVIDUAL TOTAL: 
           - If TAX/SERVICE is enabled: Person's Total = Raw Food Cost * (1 + Tax Percentage).
           - If TAX/SERVICE is disabled: Person's Total = Raw Food Cost.
        
        5. FINAL RECONCILIATION: Sum all individual totals. If the sum != Receipt Total, adjust the first person's amount by the difference (usually 0.01 or 0.02) to ensure a 100% match.
        
        OUTPUT FORMAT:
        - "Math Log": A brief explanation of the Tax % used and who was assigned what.
        - JSON ARRAY:
        [
          {{"name": "Name", "amount": 0.00, "items": "Item A x1, Item B x0.25"}}
        ]
        """.format(
            people=request.people_list,
            receipt=request.receipt_data,
            instruction=request.user_instruction or "No instructions - split everything equally.",
            tax_enabled=request.apply_tax,
            currency=curr
        )

        response = chat_model.invoke(prompt)
        return {"result": response.content}
    except Exception as e:
        print(f"Split Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))