"""Corrective retrieval: grade evidence, rewrite once, and fail closed."""
import json
import re

import ollama

from config import LLM_MODEL
from src.retriever import retrieve_documents

MIN_RELEVANT_CHUNKS = 1
MAX_REWRITES = 1


def _fallback_relevance(question, documents):
    """Cheap conservative fallback if the grader model is unavailable."""
    terms = {term.casefold() for term in re.findall(r"[A-Za-z0-9_\u0600-\u06ff]+", question) if len(term) > 2}
    relevant = []
    for document in documents:
        body = document.get("text", "").casefold()
        overlap = sum(term in body for term in terms)
        distance = document.get("distance")
        if overlap or (distance is not None and distance < 0.45):
            relevant.append({**document, "relevance": "relevant", "grade_reason": "lexical or strong semantic match"})
    return relevant


def grade_documents(question, documents):
    """Use the configured LLM to retain only chunks that answer the question."""
    if not documents:
        return []
    compact = [{"id": i, "text": d.get("text", "")[:1800]} for i, d in enumerate(documents)]
    prompt = (
        "You are a strict retrieval relevance grader. For each passage, decide whether it contains "
        "information that can help answer the question. Return JSON only as "
        '{"grades":[{"id":0,"relevant":true,"reason":"brief reason"}]}. '
        "Do not use outside knowledge. Keep a passage only when it directly supports an answer.\n"
        f"Question: {question}\nPassages: {json.dumps(compact, ensure_ascii=False)}"
    )
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0, "num_ctx": 4096},
            think=False,
            keep_alive="30m",
        )
        payload = json.loads(response["message"]["content"])
        grades = {int(row["id"]): row for row in payload.get("grades", [])}
        relevant = []
        for i, document in enumerate(documents):
            grade = grades.get(i, {})
            if grade.get("relevant") is True:
                relevant.append({**document, "relevance": "relevant", "grade_reason": str(grade.get("reason", ""))[:240]})
        return relevant
    except Exception:
        # Retrieval should remain available without letting an unavailable or
        # malformed grader turn arbitrary nearest neighbors into evidence.
        return _fallback_relevance(question, documents)


def rewrite_query(question):
    """Produce one concise search query in the student's language."""
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": (
            "Rewrite this student question as one concise, specific search query for course documents. "
            "Preserve its language and technical terms. Return only the rewritten query; do not answer it.\n"
            f"Question: {question}"
        )}],
        options={"temperature": 0, "num_ctx": 1024},
        think=False,
        keep_alive="30m",
    )
    rewritten = " ".join(response["message"]["content"].split()).strip('"` ')
    return rewritten if rewritten and rewritten.casefold() != question.casefold() else question


def corrective_retrieve(question, course_name, retriever=None, grader=None, rewriter=None):
    """Retrieve, grade, rewrite/retrieve once when weak, and return trace metadata."""
    retriever = retriever or retrieve_documents
    grader = grader or grade_documents
    rewriter = rewriter or rewrite_query
    first = retriever(question, course_name=course_name)
    accepted = grader(question, first)
    queries = [question]
    attempts = [{"query": question, "retrieved": len(first), "accepted": len(accepted)}]
    if len(accepted) < MIN_RELEVANT_CHUNKS:
        rewritten = rewriter(question)
        if rewritten and rewritten != question:
            queries.append(rewritten)
            second = retriever(rewritten, course_name=course_name)
            accepted = grader(rewritten, second)
            attempts.append({"query": rewritten, "retrieved": len(second), "accepted": len(accepted)})
    return {
        "documents": accepted,
        "queries": queries,
        "attempts": attempts,
        "corrected": len(queries) > 1,
        "sufficient": len(accepted) >= MIN_RELEVANT_CHUNKS,
    }
