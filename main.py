import os
import json
from dotenv import load_dotenv

load_dotenv()

with open("data/documents.json", encoding="utf-8") as f:
    data = json.load(f)

print("Loaded:", len(data))

provider = "openai" if os.getenv("AI_PROVIDER") == "openai" else "gemini"

print("Provider:", provider)