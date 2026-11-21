import json
from langchain_community.vectorstores import FAISS
from project.loader import load_documents
from project.embeddings import get_embeddings
from project.config import INDEX_PATH, get_provider


def get_index_provider_metadata():
    embeddings = get_embeddings()
    return {
        "provider": get_provider(),
        "embedding_model": getattr(embeddings, "model", None),
    }

def build_vectorstore():
    documents = load_documents()
    if not documents:
        raise ValueError("The academic paper dataset is empty.")

    embeddings = get_embeddings()

    db = FAISS.from_documents(documents, embeddings)
    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    db.save_local(str(INDEX_PATH))
    (INDEX_PATH / "index_metadata.json").write_text(
        json.dumps(get_index_provider_metadata(), indent=2),
        encoding="utf-8",
    )
    
    return db


def load_vectorstore():
    if not INDEX_PATH.joinpath("index.faiss").is_file():
        raise FileNotFoundError(
            "The FAISS index is missing. Build it with "
            "python -c \"from project.vectorstore import build_vectorstore; build_vectorstore()\""
        )

    metadata_path = INDEX_PATH / "index_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        current = get_index_provider_metadata()
        if metadata.get("provider") != current["provider"] or (
            metadata.get("embedding_model") and
            current["embedding_model"] and
            metadata["embedding_model"] != current["embedding_model"]
        ):
            raise ValueError(
                "The FAISS index was built with a different embedding provider or model. "
                "Rebuild the index after selecting the intended AI_PROVIDER."
            )

    return FAISS.load_local(
        str(INDEX_PATH),
        get_embeddings(),
        allow_dangerous_deserialization=False,
    )