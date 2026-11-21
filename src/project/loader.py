import csv
import math
import re
from langchain_core.documents import Document
from project.config import DATA_PATH


def normalize_text(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


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

    with open(DATA_PATH, encoding="utf-8-sig", newline="") as f:
        reader=csv.DictReader(f)
        for row in reader:
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

    return documents
