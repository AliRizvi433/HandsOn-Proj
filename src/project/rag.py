from langchain_google_genai import ChatGoogleGenerativeAI
from project.retriever import search_papers

def ask_question(query, min_year=None):
    papers = search_papers(query, min_year=min_year)

    if not papers:
        return "No matching papers found."

    context = "\n\n".join(
        f"Title: {p.metadata['title']}\nYear: {p.metadata['year']}\Content: {p.page_content}"
        for p in papers
    )

    llm = ChatGoogleGenerativeAI(
        model = "gemini-3.5-flash-lite"
    )

    prompt = f"""
You are an academic research assistant.

Answer the question using ONLY the provided papers.
Do not invent Information.
Cite Sources using paper titles and publication year.
if the papers lack sufficient information, say so.

PAPERS:
{context}

QUESTION:
{query}
"""

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):
        return "\n".join(
            block.get("text","")
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
        )

    return str(content)