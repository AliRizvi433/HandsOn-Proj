from langchain_community.vectorstores import FAISS
from project.loader import load_documents
from project.embeddings import get_embeddings

def build_vectorstore():
    documents = load_documents()
    embeddings = get_embeddings()

    db = FAISS.from_documents(documents, embeddings)
    db.save_local("data/faiss_index")
    
    return db