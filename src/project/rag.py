from dataclasses import dataclass
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from project.retriever import search_papers


@dataclass(frozen=True)
class RagResult:
    answer: str
    retrieved_papers: list
    requested_count: int
    returned_count: int
    limited_by_available_matches: bool


def ask_question(query, min_year=None, k=5, llm=None):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query must not be empty.")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")

    results = search_papers(query, min_year=min_year, k=k)
    if not results:
        return RagResult(
            answer="No matching papers were found for the current search.",
            retrieved_papers=[],
            requested_count=k,
            returned_count=0,
            limited_by_available_matches=True,
        )

    context = "\n\n".join(
        f"Title: {result.document.metadata.get('title', 'Unknown')}\n"
        f"Year: {result.publication_year if result.publication_year is not None else 'Unknown'}\n"
        f"Content: {result.document.page_content}"
        for result in results
    )

    if llm is None:
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

    prompt = f"""
You are an academic research assistant.

Answer the question using ONLY the provided paper context.
Do not make unsupported claims or invent papers, citations, or references.
Cite paper titles and publication years for claims drawn from the papers.
If the retrieved context is insufficient, state that clearly.
Distinguish your explanation from the separately displayed retrieved paper list.

PAPERS:
{context}

QUESTION:
{query}
"""

    response = llm.invoke(prompt)
    content = response.content

    if isinstance(content, list):
        answer = "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
        )
    else:
        answer = str(content)

    return RagResult(
        answer=answer,
        retrieved_papers=results,
        requested_count=k,
        returned_count=len(results),
        limited_by_available_matches=len(results) < k,
    )