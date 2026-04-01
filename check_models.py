 
import os
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(".env"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

print("Models available for content generation:\n")
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(f"  {m.name}")
        print(f"    Input limit:  {m.input_token_limit}")
        print(f"    Output limit: {m.output_token_limit}")
        print()