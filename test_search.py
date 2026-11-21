from project.retriever import search_papers

results = search_papers(
    "quantum resistant encryption",
    min_year=2023
)

for doc in results:
    print(doc.metadata["title"])
    print("Year:", doc.metadata["year"])
    print("---")