from retriever import retrieve_documents
from llm_config import load_llm

llm = load_llm()

def check_relevance(query, docs):
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Query: {query}

    Context:
    {context}

    Is the context relevant enough to answer the query?
    Reply only YES or NO.
    """

    response = llm.invoke(prompt)
    return response.content.strip()

def refine_query(query):
    prompt = f"""
    Rewrite the query to improve document retrieval:

    {query}
    """

    response = llm.invoke(prompt)
    return response.content.strip()

def generate_answer(query, docs):
    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = f"""
    Use the context below to answer the question.
    If answer is not found, say you don't know.

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(final_prompt)
    return response.content

def corrective_rag(query):
    print("🔎 Retrieving documents...")
    docs = retrieve_documents(query)

    print("🧠 Checking relevance...")
    relevance = check_relevance(query, docs)

    if "NO" in relevance.upper():
        print("⚠ Not relevant. Refining query...")
        refined_query = refine_query(query)
        docs = retrieve_documents(refined_query)

    print("✍ Generating answer...")
    answer = generate_answer(query, docs)

    return answer, docs