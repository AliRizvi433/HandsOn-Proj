import pytest
from langchain_core.documents import Document
from project.rag import ask_question
from project import rag
from project.retriever import PaperResult


class MockResponse:
    content = "The retrieved papers discuss this topic."


class MockLLM:
    def __init__(self):
        self.prompt = None

    def invoke(self, prompt):
        self.prompt = prompt
        return MockResponse()


def test_rag_passes_k_and_uses_retrieved_context(monkeypatch):
    document = Document(
        page_content="Quantum-resistant cryptography content",
        metadata={"paper_id": "P1", "title": "Quantum Cryptography", "year": 2026, "discipline": "Security"},
    )
    paper = PaperResult(document, 2026, 0.9, 1.0, 0.92)
    captured = {}

    def fake_search(query, min_year=None, k=5):
        captured.update(query=query, min_year=min_year, k=k)
        return [paper]

    monkeypatch.setattr(rag, "search_papers", fake_search)
    llm = MockLLM()

    result = ask_question("quantum-resistant cryptography", min_year=2023, k=10, llm=llm)

    assert captured == {
        "query": "quantum-resistant cryptography",
        "min_year": 2023,
        "k": 10,
    }
    assert result.requested_count == 10
    assert result.returned_count == 1
    assert result.limited_by_available_matches
    assert "Quantum Cryptography" in llm.prompt
    assert "Year: 2026" in llm.prompt


def test_rag_no_matches(monkeypatch):
    monkeypatch.setattr(rag, "search_papers", lambda *args, **kwargs: [])
    result = ask_question("machine learning", k=5)

    assert result.answer.startswith("No matching papers")
    assert result.returned_count == 0
    assert result.retrieved_papers == []


@pytest.mark.parametrize("k", [0, -1, True])
def test_rag_invalid_k(k):
    with pytest.raises(ValueError, match="positive integer"):
        ask_question("test", k=k)
