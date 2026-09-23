import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Initialize the client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("--- Available Models for your API Key ---")
try:
    # In the 2026 SDK, list() returns a simple list of model objects
    for m in client.models.list():
        # Each model 'm' has 'name' and 'display_name' directly
        print(f"ID: {m.name} | Label: {m.display_name}")
except Exception as e:
    print(f"Error: {e}")