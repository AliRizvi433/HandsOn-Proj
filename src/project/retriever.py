from dataclasses import dataclass
from typing import Optional

import faiss
from langchain_community.vectorstores import FAISS
from project.embeddings import get_embeddings
from project.config import INDEX_PATH


@dataclass(frozen=True)
class PaperResult:
    document: object
    publication_year: Optional[int]
    semantic_score: float
    recency_score: float
    combined_score: float


def _validate_k(k):
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")


def _valid_year(document):
    try:
        year = int(document.metadata.get("year", 0))
        return year if year > 0 else None
    except (TypeError, ValueError):
        return None


def _normalize_raw_scores(raw_scores, metric_type):
    """FAISS candidate scores are L2 distances or inner-product similarities."""
    if not raw_scores:
        return []

    if metric_type == faiss.METRIC_L2:
        lowest = min(raw_scores)
        highest = max(raw_scores)
        if highest == lowest:
            return [1.0] * len(raw_scores)
        return [
            1.0 - (max(0.0, float(score)) - lowest) / (highest - lowest)
            for score in raw_scores
        ]

    lowest = min(raw_scores)
    highest = max(raw_scores)
    if highest == lowest:
        return [1.0] * len(raw_scores)
    return [
        (float(score) - lowest) / (highest - lowest)
        for score in raw_scores
    ]

def search_papers(query, min_year=None, k=5):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query must not be empty.")
    _validate_k(k)

    if min_year is not None:
        if isinstance(min_year, bool):
            raise ValueError("Minimum publication year must be a valid integer.")
        try:
            min_year = int(min_year)
        except (TypeError, ValueError):
            raise ValueError("Minimum publication year must be a valid integer.")

    if not INDEX_PATH.joinpath("index.faiss").is_file():
        raise FileNotFoundError(
            "The FAISS index is missing. Rebuild it with "
            "python -c \"from project.vectorstore import build_vectorstore; build_vectorstore()\""
        )

    db = FAISS.load_local(
        str(INDEX_PATH),
        get_embeddings(),
        allow_dangerous_deserialization=False,
    )
    indexed_count = int(db.index.ntotal)
    if indexed_count <= 0:
        return []

    # Every indexed document is evaluated so filtering and deduplication remain exact.
    candidates = db.similarity_search_with_score(query, k=indexed_count)
    metric_type = getattr(db.index, "metric_type", faiss.METRIC_L2)
    raw_scores = [float(score) for _, score in candidates]
    semantic_scores = _normalize_raw_scores(raw_scores, metric_type)

    eligible_by_paper = {}
    for (doc, _), semantic_score in zip(candidates, semantic_scores):
        year = _valid_year(doc)
        if min_year is not None and (year is None or year < min_year):
            continue
        paper_id = str(doc.metadata.get("paper_id") or doc.metadata.get("title") or "").strip()
        if not paper_id:
            continue
        current = eligible_by_paper.get(paper_id)
        if current is None or semantic_score > current["semantic_score"]:
            eligible_by_paper[paper_id] = {
                "document": doc,
                "publication_year": year,
                "semantic_score": semantic_score,
            }

    if not eligible_by_paper:
        return []

    known_years = [
        item["publication_year"]
        for item in eligible_by_paper.values()
        if item["publication_year"] is not None
    ]
    if known_years:
        oldest_year = min(known_years)
        newest_year = max(known_years)
    else:
        oldest_year = newest_year = 0

    ranked = []
    for paper_id, item in eligible_by_paper.items():
        year = item["publication_year"]
        if year is None:
            recency_score = 0.0
        elif newest_year == oldest_year:
            recency_score = 1.0
        else:
            recency_score = max(0.0, min(1.0, (year - oldest_year) / (newest_year - oldest_year)))

        combined_score = 0.8 * item["semantic_score"] + 0.2 * recency_score
        ranked.append(PaperResult(
            document=item["document"],
            publication_year=year,
            semantic_score=item["semantic_score"],
            recency_score=recency_score,
            combined_score=combined_score,
        ))

    # Stage 1: select best k with relevance and recency combined.
    ranked.sort(
        key=lambda result: (
            -result.combined_score,
            str(result.document.metadata.get("paper_id", "")),
        )
    )
    selected = ranked[:k]

    # Stage 2: present selected papers newest first, with combined score as a tie-breaker.
    selected.sort(
        key=lambda result: (
            -(result.publication_year if result.publication_year is not None else -1),
            -result.combined_score,
            str(result.document.metadata.get("paper_id", "")),
        )
    )
    return selected