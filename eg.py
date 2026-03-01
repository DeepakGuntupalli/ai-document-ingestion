from corrective_chain import corrective_rag

if __name__ == "__main__":
    query = input("Enter your question: ")

    answer, docs = corrective_rag(query)

    print("\n===== FINAL ANSWER =====\n")
    print(answer)

    print("\n===== SOURCES =====")
    for doc in docs:
        print("-", doc.metadata.get("source", "Unknown"))