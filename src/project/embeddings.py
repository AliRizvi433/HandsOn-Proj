from project.config import get_provider


def get_embedding_model():
    if get_provider() == "openai":
        return "text-embedding-3-small"
    return "gemini-embedding-001"


def get_embeddings():
    if get_provider() == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=get_embedding_model(),
        )

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=get_embedding_model(),
    )