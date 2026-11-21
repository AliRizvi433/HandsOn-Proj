from langchain_community.vectorstores import FAISS
from project.embeddings import get_embeddings

def search_papers(query, min_year=None, k=5):
    db = FAISS.load_local(
        "data/faiss_index",
        get_embeddings(),
        allow_dangerous_deserialization=True
    )

    results = db.similarity_search_with_score(query, k=57)
    ranked = []

    for doc, distance in results:
        year = int(doc.metadata.get("year", 0))

        if min_year is not None and year < min_year:
            continue

        semantic_score = 1 / (1 + max(distance,0))
        recency_score = max(0, min(1,(year-2015)/11))

        final_score = (
            0.8 * semantic_score +
            0.2 * recency_score
        )

        ranked.append(((final_score, doc)))

    ranked.sort(key=lambda item: item[0], reverse=True)

    return [doc for score, doc in ranked[:k]]