from vector_store import load_vectorstore

def retrieve_documents(query, k=3):
    db = load_vectorstore()
    docs = db.similarity_search(query, k=k)
    return docs