from project.config import get_provider

def get_embeddings():
    if get_provider() == "openai":
        from langchain_openai import OpenAIEmbeddings
        
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
        )

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
    )