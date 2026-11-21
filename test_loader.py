import csv

import pytest

from project import loader


FIELD_NAMES = ["paper_id", "title", "abstract", "full_text", "keywords", "year", "discipline"]


def write_papers(path, papers):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELD_NAMES)
        writer.writeheader()
        writer.writerows(papers)


def paper(paper_id, title, abstract="Abstract", full_text="Full text", year="2025"):
    return {
        "paper_id": paper_id,
        "title": title,
        "abstract": abstract,
        "full_text": full_text,
        "keywords": "keywords",
        "year": year,
        "discipline": "Test",
    }


def test_duplicate_ids_whitespace_and_case_are_removed(monkeypatch, tmp_path, capsys):
    csv_path = tmp_path / "papers.csv"
    write_papers(csv_path, [
        paper(" P1 ", "  First   title ", "First abstract", "First full text"),
        paper("p1", "Second title", "Second abstract", "Second full text"),
        paper("P2", "Different paper", "Different abstract", "Different full text"),
    ])
    monkeypatch.setattr(loader, "DATA_PATH", csv_path)

    documents = loader.load_documents()

    assert len(documents) == 2
    assert documents[0].metadata["paper_id"] == "P1"
    assert documents[0].metadata["title"] == "First title"
    assert "Duplicate paper IDs removed: 1" in capsys.readouterr().out


def test_duplicate_content_with_different_ids_is_removed(monkeypatch, tmp_path, capsys):
    csv_path = tmp_path / "papers.csv"
    write_papers(csv_path, [
        paper("P10", "Shared  Title", " Same   abstract ", "Same full  text"),
        paper("P11", "shared title", "same abstract", "same full text"),
        paper("P12", "Distinct title", "Same abstract", "Same full text"),
    ])
    monkeypatch.setattr(loader, "DATA_PATH", csv_path)

    documents = loader.load_documents()

    assert [document.metadata["paper_id"] for document in documents] == ["P10", "P12"]
    assert "Duplicate content records removed: 1" in capsys.readouterr().out


def test_similar_but_not_identical_papers_remain(monkeypatch, tmp_path):
    csv_path = tmp_path / "papers.csv"
    write_papers(csv_path, [
        paper("A", "Quantum networks", "Same field abstract", "Text one"),
        paper("B", "Quantum network", "Same field abstract", "Text two"),
        paper("C", "Quantum networks", "Another abstract", "Text one"),
    ])
    monkeypatch.setattr(loader, "DATA_PATH", csv_path)

    documents = loader.load_documents()

    assert len(documents) == 3


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (paper("", "Title"), "without a paper_id"),
        (paper("P1", ""), "missing a title"),
        (paper("P1", "Title", year="not-a-year"), "invalid publication year"),
    ],
)
def test_existing_validation_is_preserved(monkeypatch, tmp_path, row, message):
    csv_path = tmp_path / "papers.csv"
    write_papers(csv_path, [row])
    monkeypatch.setattr(loader, "DATA_PATH", csv_path)

    with pytest.raises(ValueError, match=message):
        loader.load_documents()
