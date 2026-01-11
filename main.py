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

# 3. Security (CORS) 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
        - Total Service Tax (SST) / Service Charge
        - Total Amount of the entire bill

        Return strictly valid JSON in this format:
        {
          "items": [
            {"name": "ITEM NAME", "quantity": 1, "unit_price": 0.0, "total_price": 0.0}
          ],
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
        # PROMPT UPDATED: For professional proportional tax calculation
        prompt = f"""
        ACT AS A PROFESSIONAL ACCOUNTANT
        
        INPUTS:
        - PEOPLE: {request.people_list}
        - RECEIPT JSON: {request.receipt_data}
        - USER INSTRUCTION: {request.user_instruction}
        - APPLY TAX: {request.apply_tax}
        
        LOGIC:
        1. Assign items to people based on the instruction. If an item has quantity > 1 (e.g., 2 Craving Sets), split the quantity according to instructions or equally.
        2. Calculate the "Subtotal" for each person.
        3. If APPLY TAX is true, calculate the Tax and Service Charge proportionally. 
           (Formula: Person's Subtotal / Total Food Subtotal * Total Tax).
        4. Ensure the sum of all individual totals equals the Receipt Total.
        
        OUTPUT FORMAT:
        Write a brief "Math Log" explaining how you assigned the Craving Sets and calculated the tax.
        
        Then, at the very end, provide a JSON array strictly like this:
        [
          {{"name": "PersonName", "amount": 0.0, "items": "Item A x1, Item B x0.5"}}
        ]
        """
        response = chat_model.invoke(prompt)
        return {"result": response.content}
    except Exception as e:
        print(f"Split Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))