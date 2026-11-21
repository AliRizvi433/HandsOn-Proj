import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "academic_papers.csv"

load_dotenv(ROOT / ".env", override=False)

def get_provider():
    choice = os.getenv("AI_PROVIDER", "auto").lower()

    if choice not in ("auto", "gemini", "openai"):
        raise ValueError("Invalid AI_PROVIDER")

    if choice == "auto":
        choice = "openai" if os.getenv("OPENAI_API_KEY") else "gemini"

    key = 'OPENAI_API_KEY' if choice == "openai" else "GOOGLE_API_KEY"

    if not os.getenv(key):
        raise ValueError(f"Missing {key}")

    return choice

