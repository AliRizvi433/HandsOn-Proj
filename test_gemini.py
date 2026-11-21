import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

result = client.models.embed_content(
    model = "gemini-embedding-001",
    contents = "Redis Connection Failure"
)

print("Dimensions:", len(result.embeddings[0].values))
print("gemini embedding works")