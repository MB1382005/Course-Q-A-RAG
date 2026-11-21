
from src.retriever import retrieve_documents

question = "What is a Database Management System?"

results = retrieve_documents(question)

print("\n========== SEARCH RESULTS ==========")

for index, result in enumerate(results, start=1):

    print(f"\nResult {index}")
    print("=" * 60)

    print("Source:", result["metadata"]["source"])
    print("Page:", result["metadata"].get("page", "N/A"))
    print("Distance:", round(result["distance"], 4))

    print("\nText:")
    print(result["text"][:500])

    print("=" * 60)
