
import ollama
import re

from src.vector_store import collection
from src.course_catalog import COURSES, GENERAL_COURSE
from config import EMBEDDING_MODEL, TOP_K


def retrieve_documents(question, top_k=TOP_K, course_name="Database Management Systems"):
    question = " ".join(str(question).split())
    if not question:
        return []
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    if course_name == GENERAL_COURSE:
        course_filter = {"course": {"$in": [name for name, info in COURSES.items() if info["indexed"]]}}
    else:
        course_filter = {"course": course_name}

    # For mixed Arabic/English questions, also search using the technical
    # English term in a short English query. This helps multilingual embedding
    # models connect Arabic intent to English-only lecture slides.
    queries = [question]
    has_arabic = any("\u0600" <= char <= "\u06ff" for char in question)
    english_terms = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", question)
    if has_arabic and english_terms:
        queries.append(f"What is {' '.join(english_terms)}?")

    # 1. Convert the question (and mixed-language query when needed) to embeddings.
    embed_input = queries if len(queries) > 1 else question
    response = ollama.embed(model=EMBEDDING_MODEL, input=embed_input)

    # 2. Search each embedding in the selected course and merge candidates.
    candidates = []
    for question_embedding in response["embeddings"]:
        # Over-fetch so short slide fragments do not crowd out a longer
        # defining passage just beyond the first few nearest neighbors.
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=max(top_k * 4, top_k),
            where=course_filter
        )
        documents = (results.get("documents") or [[]])[0] or []
        metadatas = (results.get("metadatas") or [[]])[0] or []
        distances = (results.get("distances") or [[]])[0] or []
        for index, text in enumerate(documents):
            if not isinstance(text, str) or len(text.strip()) < 25:
                continue
            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            candidates.append({
                "text": text,
                "metadata": metadata,
                "distance": distances[index] if index < len(distances) else None,
            })

    # Arabic questions often include a technical term in English (for
    # example, "يعني اي Pact"). Add exact-term matches so short slide headings
    # cannot hide the page that actually defines that term.
    if has_arabic and english_terms:
        stop_words = {"what", "is", "the", "and", "for", "about", "explain", "define", "tell", "me"}
        for term in dict.fromkeys(word.upper() for word in english_terms if len(word) >= 3 and word.lower() not in stop_words):
            exact_results = collection.get(
                where=course_filter,
                where_document={"$contains": term},
                include=["documents", "metadatas"],
            )
            for text, metadata in zip(exact_results.get("documents") or [], exact_results.get("metadatas") or []):
                if isinstance(text, str) and len(text.strip()) >= 25:
                    candidates.append({
                        "text": text,
                        "metadata": metadata or {},
                        "distance": None,
                        "lexical_score": text.upper().count(term),
                    })

    # 3. Preserve relevance order and remove duplicate passages from the
    # multiple searches, returning at most top_k supporting chunks.
    candidates.sort(key=lambda item: (
        -item.get("lexical_score", 0),
        item["distance"] if item["distance"] is not None else float("inf"),
    ))
    retrieved_documents = []
    seen = set()
    for document in candidates:
        metadata = document["metadata"]
        # Chroma uses squared L2 for this collection. Keep ranking untouched;
        # suppress exact duplicate passages only.
        key = (document["text"], metadata.get("source"), metadata.get("page"))
        if key in seen:
            continue
        seen.add(key)
        retrieved_documents.append(document)
        if len(retrieved_documents) >= top_k:
            break

    return retrieved_documents
