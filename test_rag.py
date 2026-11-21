from project.rag import ask_question

answer = ask_question(
    "What are the recent advances in quantum-resistant cryptography?",
    min_year=2023
)

print(answer)