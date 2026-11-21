"""Small tool-using agent for multi-step, source-grounded course questions."""
import json
import re

import ollama

from config import LLM_MODEL
from src.corrective_rag import corrective_retrieve
from src.course_catalog import COURSES, GENERAL_COURSE

MAX_SUBQUERIES = 3


def plan_question(question, course_name):
    """Have the LLM choose retrieval subqueries and explicit analysis tools."""
    course_list = ", ".join(COURSES)
    prompt = f"""You are the planner for an Agentic RAG course assistant.
Create a short executable plan for this student question. Use no more than {MAX_SUBQUERIES} retrieval queries.
Tools allowed: retrieve (search course passages), summarize (compress retrieved evidence), compare (compare evidence for concepts), calculate (evaluate a simple arithmetic expression only when the question requires it).
Return JSON only: {{"steps":[{{"tool":"retrieve","query":"...","purpose":"..."}}],"answer_strategy":"..."}}.
Always include at least one retrieve step. Keep the student's language and technical terms in queries.
Selected course: {course_name}. Available courses: {course_list}.
Question: {question}"""
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0, "num_ctx": 2048},
            think=False,
            keep_alive="30m",
        )
        payload = json.loads(response["message"]["content"])
        steps = []
        for step in payload.get("steps", []):
            tool = step.get("tool")
            query = " ".join(str(step.get("query", "")).split())
            if tool == "retrieve" and query:
                steps.append({"tool": tool, "query": query, "purpose": str(step.get("purpose", ""))[:160]})
            elif tool in {"summarize", "compare", "calculate"}:
                steps.append({"tool": tool, "purpose": str(step.get("purpose", ""))[:160]})
        retrieval_steps = [s for s in steps if s["tool"] == "retrieve"][:MAX_SUBQUERIES]
        if not retrieval_steps:
            retrieval_steps = [{"tool": "retrieve", "query": question, "purpose": "Find course evidence"}]
        other_steps = [s for s in steps if s["tool"] != "retrieve"]
        return {"steps": retrieval_steps + other_steps[:3], "answer_strategy": str(payload.get("answer_strategy", "Answer from retrieved evidence only."))[:300], "planner": "model"}
    except Exception:
        return {"steps": [{"tool": "retrieve", "query": question, "purpose": "Find course evidence"}], "answer_strategy": "Answer from verified course excerpts only.", "planner": "fallback"}


def _safe_calculate(question):
    """Evaluate only arithmetic expressions explicitly present in the question."""
    candidates = re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?:\s*[+*/-]\s*\d+(?:\.\d+)?)+(?![\w.])", question)
    results = []
    for expression in candidates[:3]:
        try:
            if re.search(r"[^\d. +*/-]", expression):
                continue
            result = eval(expression, {"__builtins__": {}}, {})
            results.append({"expression": expression, "result": result})
        except (SyntaxError, ZeroDivisionError, TypeError, ValueError):
            continue
    return results


def _tool_summary(question, evidence, compare=False):
    if not evidence:
        return "No verified passages available for this tool step."
    bundle = [{"course": d.get("metadata", {}).get("course"), "source": d.get("metadata", {}).get("source"), "text": d.get("text", "")[:1200]} for d in evidence[:8]]
    task = "Compare the supported similarities and differences across these passages." if compare else "Summarize only the information in these passages that helps answer the question."
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": f"{task} Do not add outside facts. State when evidence is insufficient.\nQuestion: {question}\nEvidence: {json.dumps(bundle, ensure_ascii=False)}"}],
        options={"temperature": 0, "num_ctx": 3072},
        think=False,
        keep_alive="30m",
    )
    return response["message"]["content"].strip()


def run_agent(question, course_name, retriever_fn=None, grader_fn=None, planner_fn=None):
    """Plan, execute safe tools, deduplicate verified evidence and return trace."""
    planner = planner_fn or plan_question
    plan = planner(question, course_name)
    evidence_by_key = {}
    attempts = []
    for step in plan["steps"]:
        if step["tool"] != "retrieve":
            continue
        query = step["query"]
        result = corrective_retrieve(
            query,
            course_name,
            retriever=retriever_fn,
            grader=grader_fn,
        )
        attempts.append({"query": query, "corrected": result["corrected"], "accepted": len(result["documents"])})
        for document in result["documents"]:
            metadata = document.get("metadata") or {}
            key = (document.get("text"), metadata.get("source"), metadata.get("page"), metadata.get("course"))
            evidence_by_key[key] = document

    evidence = list(evidence_by_key.values())
    tool_outputs = []
    for step in plan["steps"]:
        tool = step["tool"]
        if tool == "summarize":
            tool_outputs.append({"tool": tool, "purpose": step.get("purpose", ""), "output": _tool_summary(question, evidence)})
        elif tool == "compare":
            tool_outputs.append({"tool": tool, "purpose": step.get("purpose", ""), "output": _tool_summary(question, evidence, compare=True)})
        elif tool == "calculate":
            tool_outputs.append({"tool": tool, "purpose": step.get("purpose", ""), "output": _safe_calculate(question)})

    sufficient = bool(evidence)
    return {
        "documents": evidence,
        "plan": plan,
        "attempts": attempts,
        "tool_outputs": tool_outputs,
        "sufficient": sufficient,
        "agentic": True,
    }
