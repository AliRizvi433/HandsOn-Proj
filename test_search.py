import pytest
from langchain_core.documents import Document
from project import retriever


def paper(paper_id, title, year, discipline="Test"):
    return Document(
        page_content=f"{title} test content",
        metadata={
            "paper_id": paper_id,
            "title": title,
            "year": year,
            "discipline": discipline,
        },
    )


def mock_faiss(monkeypatch, tmp_path, documents_with_scores):
    class MockIndex:
        ntotal = len(documents_with_scores)
        metric_type = retriever.faiss.METRIC_L2

    class MockDB:
        index = MockIndex()

        def similarity_search_with_score(self, query, k):
            assert k == len(documents_with_scores)
            return documents_with_scores

    monkeypatch.setattr(retriever, "load_vectorstore", lambda: MockDB())


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_invalid_k(k):
    with pytest.raises(ValueError, match="positive integer"):
        retriever._validate_k(k)


def test_combined_ranking_and_newest_first(monkeypatch, tmp_path):
    documents = [
        (paper("P1", "Quantum cryptography", 2022), 0.1),
        (paper("P2", "Quantum-resistant systems", 2026), 0.5),
        (paper("P3", "Cryptography survey", 2024), 0.2),
    ]
    mock_faiss(monkeypatch, tmp_path, documents)

    results = retriever.search_papers("quantum-resistant cryptography", k=3)

    assert len(results) == 3
    assert [result.publication_year for result in results] == [2026, 2024, 2022]
    assert all(
        0.0 <= result.semantic_score <= 1.0
        and 0.0 <= result.recency_score <= 1.0
        and 0.0 <= result.combined_score <= 1.0
        for result in results
    )


def test_year_filter_and_insufficient_matches(monkeypatch, tmp_path):
    documents = [
        (paper("P1", "Machine learning 2026", 2026), 0.1),
        (paper("P2", "Machine learning 2023", 2023), 0.2),
        (paper("P3", "Machine learning 2021", 2021), 0.3),
    ]
    mock_faiss(monkeypatch, tmp_path, documents)

    results = retriever.search_papers("machine learning", min_year=2023, k=10)

    assert len(results) == 2
    assert all(result.publication_year >= 2023 for result in results)
    assert [result.publication_year for result in results] == [2026, 2023]


def test_duplicate_chunks_are_deduplicated(monkeypatch, tmp_path):
    documents = [
        (paper("P1", "Chunk one", 2025), 0.1),
        (paper("P1", "Chunk one", 2025), 0.2),
        (paper("P2", "Chunk two", 2024), 0.3),
    ]
    mock_faiss(monkeypatch, tmp_path, documents)

    results = retriever.search_papers("quantum", k=2)

    assert len(results) == 2
    assert len({result.document.metadata["paper_id"] for result in results}) == 2


@pytest.mark.parametrize(("k", "expected"), [(5, 5), (10, 10)])
def test_custom_and_default_top_k(monkeypatch, tmp_path, k, expected):
    documents = [
        (paper(f"P{i}", f"Quantum topic {i}", 2020 + (i % 7)), 0.1 + (i * 0.01))
        for i in range(12)
    ]
    mock_faiss(monkeypatch, tmp_path, documents)

    results = retriever.search_papers("quantum", k=k)

    assert len(results) == expected
    assert len({result.document.metadata["paper_id"] for result in results}) == expected


@pytest.mark.parametrize("query", ["", "   ", None])
def test_empty_query(query):
    with pytest.raises(ValueError, match="empty"):
        retriever.search_papers(query)


def test_identical_faiss_scores_are_normalized_deterministically():
    assert retriever._normalize_raw_scores([0.5, 0.5], retriever.faiss.METRIC_L2) == [1.0, 1.0]
