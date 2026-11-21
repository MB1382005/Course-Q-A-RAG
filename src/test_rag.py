
from src.retriever import retrieve_documents
from src.generator import generate_answer


questions = [
    "What is a Database Management System?",
    "ما الفرق بين Database و DBMS؟",
    "What is Quantum Computing?"
]


for question in questions:

    print("\n" + "=" * 70)
    print("QUESTION:", question)

    # 1. Retrieve relevant documents
    documents = retrieve_documents(question)

    # 2. Generate answer
    answer = generate_answer(question, documents)

    print("\nAI ANSWER:")
    print(answer)

    # 3. Display actual retrieved sources
    print("\nSOURCES:")

    for document in documents:

        metadata = document["metadata"]

        print(
            f"- {metadata.get('source')} "
            f"(Page {metadata.get('page', 'N/A')})"
        )

    print("=" * 70)
