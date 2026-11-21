import csv
from langchain_core.documents import Document
from project.config import DATA_PATH

def load_documents():
    documents = []

    with open(DATA_PATH, encoding="utf-8-sig", newline="") as f:
        reader=csv.DictReader(f)
        for row in reader:
            content = (
                f"Title: {row['title']}\n"
                f"Abstract: {row['abstract']}\n"
                f"Full Text: {row['full_text']}\n"
                f"Keywords: {row['keywords']}"
            )

            documents.append(Document(
                page_content=content,
                metadata={
                    "paper_id":row["paper_id"],
                    "title":row["title"],
                    "year":row["year"],
                    "discipline":row["discipline"],
                }
            ))

    return documents
