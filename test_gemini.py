import os
import pytest
from dotenv import load_dotenv
from google import genai

load_dotenv()


@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY is not configured")
def test_gemini_embedding():
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents="Redis Connection Failure",
    )

    assert result.embeddings
    assert result.embeddings[0].values