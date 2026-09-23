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

# 3. CORS CONFIGURATION
origins = [
    "http://localhost:3000",
    "https://billa-rho.vercel.app",      # Your exact Vercel URL
    "https://billa-rho.vercel.app/"      # Trailing slash version
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,             # usage of the list above
    allow_credentials=True,
    allow_methods=["*"],               # Allow ALL methods (POST, GET, OPTIONS)
    allow_headers=["*"],               # Allow ALL headers
)
# --------------------------

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

class ChatRequest(BaseModel):
    receipt_data: str
    history: list 
    user_message: str

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

        # UPDATED PROMPT: Enforces single JSON object output to prevent parsing errors
        prompt = """
        ACT AS A SENIOR AUDITOR. Calculate exactly how much each person owes.

        INPUTS:
        - PEOPLE: {people}
        - RECEIPT: {receipt}
        - INSTRUCTIONS: {instruction}
        - TAX/SVC INCLUDED: {tax_enabled}
        
        ALGORITHM:
        1. Parse Instructions: If an item is assigned to a person, they pay 100%. If not mentioned, split equally among ALL.
        2. Calculate Ratios: Determine the tax/service charge ratio based on the receipt totals.
        3. Distribute: Apply the tax ratio to each person's raw food cost.
        4. Reconcile: Ensure the sum of individual totals matches the Receipt Total exactly (adjust pennies on the first person if needed).

        OUTPUT FORMAT:
        Return ONLY valid JSON. Do not use Markdown blocks.
        {{
            "reasoning": "Brief log of the calculation steps and tax ratio used...",
            "splits": [
                {{"name": "Person Name", "amount": 0.00, "items": "Item A (x1), Item B (x0.5)"}}
            ]
        }}
        """.format(
            people=request.people_list,
            receipt=request.receipt_data,
            instruction=request.user_instruction or "Split everything equally.",
            tax_enabled=request.apply_tax,
            currency=curr
        )

        response = chat_model.invoke(prompt)
        return {"result": response.content}
    except Exception as e:
        print(f"Split Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat_modify")
async def chat_modify_bill(request: ChatRequest):
    try:
        # Construct a context-aware prompt
        prompt = """
        You are a helpful AI Bill Assistant. You are modifying an existing bill split based on the user's request.
        
        CONTEXT:
        - RECEIPT DATA: {receipt}
        - CHAT HISTORY: {history}
        - USER REQUEST: "{message}"

        INSTRUCTIONS:
        1. Read the User Request and update the split calculations accordingly.
        2. Keep the math precise.
        3. Be friendly in your text response.

        OUTPUT FORMAT:
        Return ONLY valid JSON. Do not use Markdown blocks.
        {{
            "reply": "Text response to the user (e.g., 'Sure, I've removed the tax for Tom.')",
            "splits": [
                {{"name": "Person Name", "amount": 0.00, "items": "Item A (x1)..."}}
            ]
        }}
        """.format(
            receipt=request.receipt_data,
            history=request.history,
            message=request.user_message
        )

        response = chat_model.invoke(prompt)
        clean_json = response.content.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
        
    except Exception as e:
        print(f"Chat Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))