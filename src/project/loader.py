import csv
import math
import re
from langchain_core.documents import Document
from project.config import DATA_PATH


def normalize_text(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_comparison_text(value):
    """Normalize text for duplicate comparisons only."""
    return normalize_text(value).casefold()


def estimate_token_count(text):
    """Conservatively estimate model tokens without adding a tokenizer dependency.

    The embedding provider performs the authoritative tokenization internally.
    This estimate is diagnostic only and is deliberately not used to reject a
    valid paper or to create chunks that could be counted as separate papers.
    """
    if not text:
        return 0
    word_piece_estimate = len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))
    character_estimate = math.ceil(len(text.encode("utf-8")) / 4)
    return max(word_piece_estimate, character_estimate)


def load_documents():
    documents = []
    total_records = 0
    duplicate_ids = 0
    duplicate_content = 0
    seen_paper_ids = set()
    seen_content = set()

    with open(DATA_PATH, encoding="utf-8-sig", newline="") as f:
        reader=csv.DictReader(f)
        for row in reader:
            total_records += 1
            paper_id = normalize_text(row.get("paper_id"))
            title = normalize_text(row.get("title"))
            abstract = normalize_text(row.get("abstract"))
            full_text = normalize_text(row.get("full_text"))
            keywords = normalize_text(row.get("keywords"))
            discipline = normalize_text(row.get("discipline"))

            if not paper_id:
                raise ValueError("Academic paper dataset contains a row without a paper_id.")
            if not title:
                raise ValueError(f"Academic paper {paper_id} is missing a title.")

            year_text = normalize_text(row.get("year"))
            try:
                year = int(year_text)
            except (TypeError, ValueError):
                raise ValueError(f"Academic paper {paper_id} has an invalid publication year: {year_text!r}")

            normalized_paper_id = normalize_comparison_text(paper_id)
            if normalized_paper_id in seen_paper_ids:
                duplicate_ids += 1
                continue

            content_key = (
                normalize_comparison_text(title),
                normalize_comparison_text(abstract),
                normalize_comparison_text(full_text),
            )
            if content_key in seen_content:
                duplicate_content += 1
                continue

            seen_paper_ids.add(normalized_paper_id)
            seen_content.add(content_key)

            content = (
                f"Title: {title}\n"
                f"Abstract: {abstract or 'Not provided'}\n"
                f"Full Text: {full_text or 'Not provided'}\n"
                f"Keywords: {keywords or 'Not provided'}"
            )
            documents.append(Document(
                page_content=content,
                metadata={
                    "paper_id": paper_id,
                    "title": title,
                    "year": year,
                    "discipline": discipline,
                }
            ))

    print(f"Total records loaded: {total_records}")
    print(f"Duplicate paper IDs removed: {duplicate_ids}")
    print(f"Duplicate content records removed: {duplicate_content}")
    print(f"Unique papers for indexing: {len(documents)}")
    return documents
