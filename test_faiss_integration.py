from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from project import retriever, vectorstore


class DeterministicEmbeddings(Embeddings):
    def _embed(self, text):
        lowered = text.lower()
        return [
            float("quantum" in lowered),
            float("machine" in lowered),
            float("learning" in lowered),
        ]

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)


def paper(paper_id, title, year, content):
    return Document(
        page_content=content,
        metadata={"paper_id": paper_id, "title": title, "year": year, "discipline": "Test"},
    )


def configure_deterministic_index(monkeypatch, tmp_path, documents):
    monkeypatch.setattr(vectorstore, "INDEX_PATH", tmp_path / "faiss_index")
    monkeypatch.setattr(vectorstore, "get_provider", lambda: "gemini")
    monkeypatch.setattr(vectorstore, "get_embedding_model", lambda: "deterministic-test")
    monkeypatch.setattr(vectorstore, "get_embeddings", lambda: DeterministicEmbeddings())
    monkeypatch.setattr(vectorstore, "load_documents", lambda: documents)
    monkeypatch.setattr(retriever, "load_vectorstore", vectorstore.load_vectorstore)


def test_real_faiss_index_is_saved_loaded_and_retrieved(monkeypatch, tmp_path):
    documents = [
        paper("P1", "Quantum security", 2026, "quantum cryptography"),
        paper("P2", "Machine learning", 2024, "machine learning"),
    ]
    configure_deterministic_index(monkeypatch, tmp_path, documents)

    vectorstore.build_vectorstore()
    loaded = vectorstore.load_vectorstore()

    assert (tmp_path / "faiss_index" / "index.faiss").is_file()
    assert (tmp_path / "faiss_index" / "index.pkl").is_file()
    assert (tmp_path / "faiss_index" / "index_metadata.json").is_file()
    assert loaded.index.ntotal == 2
    assert vectorstore.get_distinct_paper_count(loaded) == 2

    results = retriever.search_papers("quantum", min_year=2025, k=2)

    assert len(results) == 1
    assert results[0].document.metadata["paper_id"] == "P1"
    assert results[0].publication_year == 2026


def test_index_without_metadata_requires_rebuild(monkeypatch, tmp_path):
    documents = [paper("P1", "Quantum security", 2026, "quantum cryptography")]
    configure_deterministic_index(monkeypatch, tmp_path, documents)
    vectorstore.build_vectorstore()
    (tmp_path / "faiss_index" / "index_metadata.json").unlink()

    try:
        vectorstore.load_vectorstore()
    except ValueError as exc:
        assert "index_metadata.json" in str(exc)
        assert "build_vectorstore" in str(exc)
    else:
        raise AssertionError("An index without metadata must not load")


def test_incompatible_index_metadata_requires_rebuild(monkeypatch, tmp_path):
    documents = [paper("P1", "Quantum security", 2026, "quantum cryptography")]
    configure_deterministic_index(monkeypatch, tmp_path, documents)
    vectorstore.build_vectorstore()
    (tmp_path / "faiss_index" / "index_metadata.json").write_text(
        '{"provider": "openai", "embedding_model": "text-embedding-3-small"}',
        encoding="utf-8",
    )

    try:
        vectorstore.load_vectorstore()
    except ValueError as exc:
        assert "different embedding provider or model" in str(exc)
        assert "build_vectorstore" in str(exc)
    else:
        raise AssertionError("An incompatible index must not load")
