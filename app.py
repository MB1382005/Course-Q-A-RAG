import json
import re
import time
import uuid
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import streamlit as st

from config import BASE_DIR, EMBEDDING_MODEL, LLM_MODEL
from src.agentic_rag import run_agent
from src.corrective_rag import corrective_retrieve
from src.course_catalog import COURSES, GENERAL_COURSE
from src.generator import generate_answer
from src.index_course import index_course
from src.retriever import retrieve_documents
from src.vector_store import collection
from src.zip_ingestion import extract_course_zip

CHAT_HISTORY_PATH = BASE_DIR / "chat_history.json"
RAG_MODES = ["Corrective RAG", "Standard RAG", "Agentic RAG"]
DEFAULT_RAG_MODE = "Corrective RAG"
COURSE_OPTIONS = [GENERAL_COURSE, *COURSES.keys()]


def _is_arabic(text: str) -> bool:
    return any("\u0600" <= char <= "\u06ff" for char in str(text))


def text_direction(text: str) -> str:
    arabic_count = sum("\u0600" <= char <= "\u06ff" for char in text)
    latin_count = sum(char.isascii() and char.isalpha() for char in text)
    return "rtl" if arabic_count > latin_count else "ltr"


def _display_filename(filename: str) -> str:
    """Strip legacy 32-char hex OCW download prefix for cleaner display."""
    name = str(filename)
    if len(name) > 33 and name[32] == "_" and all(c in "0123456789abcdef" for c in name[:32].lower()):
        return name[33:]
    return name


def collect_sources(documents):
    """Return unique source/page pairs present in retrieved metadata."""
    sources, seen = [], set()
    for document in documents:
        metadata = document.get("metadata") or {}
        # Accept raw Chroma query/get records as well as the normalized
        # retriever shape. This keeps citation extraction tied to actual
        # stored metadata even if a caller passes a Chroma result directly.
        if not metadata and isinstance(document, dict):
            metadata = document
        filename = metadata.get("source")
        if not isinstance(filename, str) or not filename.strip():
            continue
        page = metadata.get("page")
        if not isinstance(page, int) or page < 1:
            page = None
        course = metadata.get("course")
        course = course.strip() if isinstance(course, str) and course.strip() else None
        key = (filename, page, course)
        if key not in seen:
            entry = {"file": filename, "page": page}
            if course:
                entry["course"] = course
            sources.append(entry)
            seen.add(key)
    return sources


def _format_message_html(text: str, direction: str) -> str:
    safe = escape(str(text), quote=False)
    if direction == "rtl":
        # Preserve HTML entities (&amp;, &lt;, &gt;) untouched while isolating
        # English technical terms before inserting any HTML tags.
        def _wrap_ltr(match):
            if match.group(1):
                return match.group(1)
            return f"<span class='ltr-term'>{match.group(2)}</span>"

        safe = re.sub(
            r"(&(?:[A-Za-z]+|#\d+|#x[0-9A-Fa-f]+);)|([A-Za-z][A-Za-z0-9_./:+-]*(?: [A-Za-z0-9_./:+-]+){0,6})",
            _wrap_ltr,
            safe,
        )
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
    safe = safe.replace("\n", "<br>")
    return (
        f"<div class='message-content answer-content' dir='{direction}' "
        f"style='text-align:{'right' if direction == 'rtl' else 'left'}'>{safe}</div>"
    )


def _now_timestamp() -> float:
    return time.time()


def _timestamp_value(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0
    return 0.0


def _new_chat_record(course: str = GENERAL_COURSE, rag_mode: str = DEFAULT_RAG_MODE) -> dict:
    return {
        "title": "New chat",
        "messages": [],
        "course": course if course in COURSE_OPTIONS else GENERAL_COURSE,
        "rag_mode": rag_mode if rag_mode in RAG_MODES else DEFAULT_RAG_MODE,
        "updated_at": _now_timestamp(),
    }


def load_chat_store() -> dict:
    if CHAT_HISTORY_PATH.exists():
        try:
            data = json.loads(CHAT_HISTORY_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("chats"), dict) and data["chats"]:
                for chat in data["chats"].values():
                    chat.setdefault("title", "New chat")
                    chat.setdefault("messages", [])
                    if chat.get("course") not in COURSE_OPTIONS:
                        chat["course"] = GENERAL_COURSE
                    if chat.get("rag_mode") not in RAG_MODES:
                        chat["rag_mode"] = DEFAULT_RAG_MODE
                    chat["updated_at"] = _timestamp_value(chat.get("updated_at")) or _now_timestamp()
                if data.get("active_chat_id") not in data["chats"]:
                    data["active_chat_id"] = next(iter(data["chats"]))
                return data
        except Exception:
            pass
    chat_id = uuid.uuid4().hex
    return {
        "active_chat_id": chat_id,
        "course_default_migrated": True,
        "chats": {chat_id: _new_chat_record()},
    }


def save_chat_store(store: dict) -> None:
    try:
        CHAT_HISTORY_PATH.write_text(
            json.dumps(store, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _suggest_other_course(question: str, current_course: str):
    """If the selected course has no accepted evidence, check other courses for a strong match."""
    if current_course == GENERAL_COURSE:
        return None
    best_course, best_distance = None, None
    for name, info in COURSES.items():
        if name == current_course or not info.get("indexed"):
            continue
        try:
            docs = retrieve_documents(question, top_k=2, course_name=name)
        except Exception:
            continue
        if not docs:
            continue
        top = docs[0]
        lexical = top.get("lexical_score", 0)
        distance = top.get("distance")
        if lexical > 0:
            return name
        if distance is not None and distance < 0.72 and (best_distance is None or distance < best_distance):
            best_distance = distance
            best_course = name
    return best_course


def _insufficient_message(question: str, suggested_course=None) -> str:
    if _is_arabic(question):
        base = "لا تتوفر معلومات كافية في مواد المقرر المحدد للإجابة عن هذا السؤال."
        if suggested_course:
            base += f"\n\nيبدو أن السؤال أقرب إلى مقرر **{suggested_course}**. يمكنك التبديل إليه أو اختيار **{GENERAL_COURSE}** من القائمة الجانبية."
        return base
    base = "The available materials for the selected course do not provide enough information to answer this question."
    if suggested_course:
        base += f"\n\nThis question appears to match **{suggested_course}**. Try switching to that course or selecting **{GENERAL_COURSE}** in the sidebar."
    return base


def _build_reasoning(question: str, documents: list, rag_mode: str, trace: dict) -> dict:
    arabic = _is_arabic(question)
    top_doc = documents[0] if documents else None
    meta = (top_doc or {}).get("metadata") or {}
    excerpt = ((top_doc or {}).get("text") or "").strip()[:600]

    if rag_mode == "Standard RAG":
        if documents:
            summary = (
                "وضع Standard RAG: الإجابة مبنية مباشرةً على المقاطع الأقرب دلاليًا من مواد المقرر أدناه."
                if arabic
                else "Standard RAG: Retrieved nearest course passages directly without grading or query rewriting."
            )
        else:
            summary = (
                "وضع Standard RAG: لم يتم العثور على مقاطع مناسبة في مواد المقرر المحدد."
                if arabic
                else "Standard RAG: No matching passages were retrieved from the selected course."
            )
    elif rag_mode == "Corrective RAG":
        corrected = trace.get("corrected", False)
        if documents:
            if arabic:
                prefix = "تمت إعادة صياغة الاستعلام مرة واحدة وتقييم المقاطع المسترجعة. " if corrected else "اجتازت المقاطع المسترجعة فحص الصلة بالموضوع. "
                summary = prefix + "الإجابة مبنية على المقطع المسترجع أدناه وتقتصر على ما يدعمه."
            else:
                prefix = "Query correction (rewrite) was used and retrieved passages passed relevance grading. " if corrected else "Retrieved passages passed relevance grading. "
                summary = prefix + "This answer is based on the verified course passage below."
        else:
            summary = (
                "لم تجتز أي مقاطع فحص الصلة حتى بعد محاولة إعادة صياغة السؤال؛ لذلك تم الامتناع عن التوليد منعًا للهلوسة."
                if arabic
                else "No passages passed relevance grading (even after query rewrite), so answer generation was skipped."
            )
    else:
        plan = trace.get("plan") or {}
        planner = plan.get("planner", "model")
        tools = ", ".join(step.get("tool", "") for step in plan.get("steps", [])) or "retrieve"
        attempts_count = len(trace.get("attempts") or [])
        summary = (
            f"Agentic RAG ({planner}): نفّذ الخطة ({tools}) عبر {attempts_count} خطوة بحث واحتفظ بـ {len(documents)} مقطع موثّق."
            if arabic
            else f"Agent plan ({planner}): {tools}. Executed {attempts_count} retrieval step(s); retained {len(documents)} verified passage(s)."
        )

    reasoning = {
        "summary": summary,
        "excerpt": excerpt,
        "source": meta.get("source"),
        "page": meta.get("page"),
        "course": meta.get("course"),
        "mode": rag_mode,
    }
    if trace.get("tool_outputs"):
        reasoning["tool_outputs"] = trace["tool_outputs"]
    return reasoning


def render_sources(sources):
    if not sources:
        return
    with st.expander("\U0001F4DA Sources"):
        rows = []
        for source in sources:
            filename = escape(_display_filename(source["file"]))
            page = f"Page {source['page']}" if source.get("page") else "Page unavailable"
            course_html = (
                f"<div class='source-course'>{escape(source['course'])}</div>"
                if source.get("course")
                else ""
            )
            rows.append(
                f"<div class='source-row'><div class='source-icon'>\U0001F4C4</div>"
                f"<div><div class='source-name'>{filename}</div>"
                f"<div class='source-page'>{page}</div>{course_html}</div></div>"
            )
        st.markdown(f"<div class='source-list'>{''.join(rows)}</div>", unsafe_allow_html=True)


def render_reasoning(reasoning, trace=None):
    if not reasoning and not trace:
        return
    with st.expander("\U0001F4A1 Why this answer?"):
        summary = (reasoning or {}).get("summary")
        if summary:
            direction = text_direction(summary)
            st.markdown(
                f"<div class='reasoning-summary' dir='{direction}'>{escape(summary)}</div>",
                unsafe_allow_html=True,
            )

        if trace:
            plan = trace.get("plan")
            if plan:
                strategy = plan.get("answer_strategy")
                if strategy:
                    st.caption(f"Plan strategy ({plan.get('planner', 'model')}): {strategy}")
            attempts = trace.get("attempts") or []
            if attempts:
                lines = []
                for idx, attempt in enumerate(attempts, start=1):
                    q = attempt.get("query", "")
                    acc = attempt.get("accepted", 0)
                    extra = []
                    if "retrieved" in attempt:
                        extra.append(f"retrieved {attempt['retrieved']}")
                    extra.append(f"accepted {acc}")
                    if attempt.get("corrected"):
                        extra.append("query rewritten")
                    lines.append(f"{idx}. `{q}` ({', '.join(extra)})")
                st.markdown("**Retrieval trace:**\n" + "\n".join(lines))

        tool_outputs = (reasoning or {}).get("tool_outputs") or (trace or {}).get("tool_outputs") or []
        if tool_outputs:
            st.markdown("**Agent tool outputs:**")
            for item in tool_outputs:
                tool_name = item.get("tool", "tool")
                purpose = item.get("purpose", "")
                output = item.get("output")
                header = f"**{tool_name}**" + (f" — {purpose}" if purpose else "")
                st.markdown(header)
                if isinstance(output, list):
                    if output:
                        for row in output:
                            st.code(f"{row.get('expression')} = {row.get('result')}")
                    else:
                        st.caption("No safe arithmetic expression found in the question.")
                else:
                    st.info(str(output))

        excerpt = (reasoning or {}).get("excerpt")
        if excerpt:
            meta_bits = []
            if reasoning.get("course"):
                meta_bits.append(str(reasoning["course"]))
            if reasoning.get("source"):
                meta_bits.append(_display_filename(reasoning["source"]))
            if reasoning.get("page"):
                meta_bits.append(f"Page {reasoning['page']}")
            if meta_bits:
                st.caption("Top supporting excerpt: " + " · ".join(meta_bits))
            st.markdown(
                f"<div class='reasoning-excerpt'>{escape(excerpt).replace(chr(10), '<br>')}</div>",
                unsafe_allow_html=True,
            )


def render_metrics(timings):
    metrics = [
        ("\U0001F50E", "Search", timings["retrieval"]),
        ("\U0001F9E0", "AI generation", timings["generation"]),
        ("\u23F1\ufe0f", "Total response", timings["total"]),
    ]
    cards = []
    for icon, label, value in metrics:
        cards.append(
            f"<div class='metric-card'><span class='metric-icon'>{icon}</span>"
            f"<div><div class='metric-label'>{label}</div>"
            f"<div class='metric-value'>{value:.2f}s</div></div></div>"
        )
    st.markdown(f"<div class='metric-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def run_rag_pipeline(question: str, course_name: str, rag_mode: str):
    """Execute Standard, Corrective, or Agentic RAG and return answer + trace metadata."""
    retrieval_start = time.perf_counter()
    trace = {"mode": rag_mode}
    documents = []
    sufficient = True

    if rag_mode == "Standard RAG":
        documents = retrieve_documents(question, course_name=course_name)
        trace["attempts"] = [{"query": question, "retrieved": len(documents), "accepted": len(documents)}]
        sufficient = bool(documents)
    elif rag_mode == "Corrective RAG":
        result = corrective_retrieve(question, course_name=course_name)
        documents = result["documents"]
        trace = result
        sufficient = result["sufficient"]
    else:
        result = run_agent(question, course_name=course_name)
        documents = result["documents"]
        trace = result
        sufficient = result["sufficient"]

    retrieval_time = time.perf_counter() - retrieval_start

    generation_start = time.perf_counter()
    if sufficient and documents:
        answer = generate_answer(question, documents)
    else:
        suggested = _suggest_other_course(question, course_name) if rag_mode != "Standard RAG" else None
        answer = _insufficient_message(question, suggested_course=suggested)
        documents = []
    generation_time = time.perf_counter() - generation_start

    sources = collect_sources(documents)
    reasoning = _build_reasoning(question, documents, rag_mode, trace)
    timings = {
        "retrieval": retrieval_time,
        "generation": generation_time,
        "total": retrieval_time + generation_time,
    }
    return answer, sources, reasoning, trace, timings


def _course_banner_info(selected_course: str):
    if selected_course == GENERAL_COURSE:
        return "ALL", "General — All Courses", "Searching across Database Systems, HCI, and Computer Security"
    info = COURSES[selected_course]
    return info["mark"], info["name"], info["subtitle"]


def main():
    st.set_page_config(
        page_title="Course Q&A",
        page_icon="\U0001F393",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    css_path = Path(__file__).with_name("course_qa.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False
    st.markdown(
        "<div class='theme-dark'></div>" if st.session_state.dark_mode else "<div class='theme-light'></div>",
        unsafe_allow_html=True,
    )

    if "chat_store" not in st.session_state:
        st.session_state.chat_store = load_chat_store()

    store = st.session_state.chat_store
    active_id = store["active_chat_id"]
    if active_id not in store["chats"]:
        active_id = next(iter(store["chats"]))
        store["active_chat_id"] = active_id
    active_chat = store["chats"][active_id]

    try:
        total_chunks = collection.count()
    except Exception:
        total_chunks = 0

    with st.sidebar:
        st.markdown("## Course Q&A")
        st.caption("Your course-grounded study companion")
        st.toggle("Dark mode", key="dark_mode")
        st.markdown("---")

        # Course selector
        st.markdown("### Course")
        current_course = active_chat.get("course", GENERAL_COURSE)
        if current_course not in COURSE_OPTIONS:
            current_course = GENERAL_COURSE
        selected_course = st.selectbox(
            "Select course",
            COURSE_OPTIONS,
            index=COURSE_OPTIONS.index(current_course),
            label_visibility="collapsed",
            key=f"course_select_{active_id}",
        )
        if selected_course != active_chat.get("course"):
            active_chat["course"] = selected_course
            active_chat["updated_at"] = _now_timestamp()
            save_chat_store(store)

        _, course_title, course_subtitle = _course_banner_info(selected_course)
        st.info(f"**{course_title}**\n\n{course_subtitle}")

        # RAG Mode selector
        st.markdown("### RAG mode")
        current_mode = active_chat.get("rag_mode", DEFAULT_RAG_MODE)
        if current_mode not in RAG_MODES:
            current_mode = DEFAULT_RAG_MODE
        selected_mode = st.selectbox(
            "Select RAG mode",
            RAG_MODES,
            index=RAG_MODES.index(current_mode),
            label_visibility="collapsed",
            key=f"rag_mode_{active_id}",
        )
        if selected_mode != active_chat.get("rag_mode"):
            active_chat["rag_mode"] = selected_mode
            active_chat["updated_at"] = _now_timestamp()
            save_chat_store(store)

        mode_help = {
            "Standard RAG": "Direct semantic search + grounded generation (fastest).",
            "Corrective RAG": "Grades retrieved chunks and rewrites query once if weak.",
            "Agentic RAG": "Plans up to 3 sub-queries + summarize, compare & safe calculate tools.",
        }
        st.caption(mode_help[selected_mode])
        st.markdown("---")

        # Chat session manager
        st.markdown("### Conversations")
        col_new, col_del = st.columns(2)
        with col_new:
            if st.button("New chat", use_container_width=True):
                new_id = uuid.uuid4().hex
                store["chats"][new_id] = _new_chat_record(
                    course=GENERAL_COURSE,
                    rag_mode=selected_mode,
                )
                store["active_chat_id"] = new_id
                save_chat_store(store)
                st.rerun()
        with col_del:
            if st.button("Clear chat", use_container_width=True):
                active_chat["messages"] = []
                active_chat["title"] = "New chat"
                active_chat["updated_at"] = _now_timestamp()
                save_chat_store(store)
                st.rerun()

        ordered_chat_ids = sorted(
            store["chats"].keys(),
            key=lambda cid: _timestamp_value(store["chats"][cid].get("updated_at")),
            reverse=True,
        )

        def _chat_label(cid: str) -> str:
            chat = store["chats"][cid]
            title = (chat.get("title") or "New chat").strip()
            count = len(chat.get("messages") or [])
            return f"{title[:34]} ({count})" if count else f"{title[:34]}"

        chosen_chat_id = st.selectbox(
            "Switch conversation",
            ordered_chat_ids,
            index=ordered_chat_ids.index(active_id),
            format_func=_chat_label,
            label_visibility="collapsed",
        )
        if chosen_chat_id != active_id:
            store["active_chat_id"] = chosen_chat_id
            save_chat_store(store)
            st.rerun()

        if len(store["chats"]) > 1:
            if st.button("Delete current chat", use_container_width=True):
                del store["chats"][active_id]
                store["active_chat_id"] = next(iter(store["chats"]))
                save_chat_store(store)
                st.rerun()

        st.markdown("---")

        # Upload course ZIP
        with st.expander("Upload course ZIP"):
            upload_course = st.selectbox(
                "Destination course",
                list(COURSES.keys()),
                key="zip_target_course",
            )
            uploaded_zip = st.file_uploader(
                "Choose a .zip archive",
                type=["zip"],
                key="zip_uploader",
            )
            if st.button("Extract and index ZIP", use_container_width=True, disabled=uploaded_zip is None):
                try:
                    with st.spinner("Extracting and indexing course archive..."):
                        dest_folder, file_count = extract_course_zip(
                            uploaded_zip.getvalue(),
                            uploaded_zip.name,
                            COURSES[upload_course]["folder"],
                        )
                        chunk_count = index_course(dest_folder, upload_course)
                    st.success(f"Extracted {file_count} file(s) and indexed {chunk_count} chunk(s) into {upload_course}.")
                except Exception as error:
                    st.error(f"ZIP ingestion failed: {error}")

        st.markdown("---")
        st.markdown("### Study tools")
        st.markdown("<span class='sidebar-label'>Language</span><br><strong>Arabic + English</strong>", unsafe_allow_html=True)
        st.markdown(f"<br><span class='sidebar-label'>Answer model</span><br><strong>{escape(LLM_MODEL)}</strong>", unsafe_allow_html=True)
        st.markdown(f"<br><span class='sidebar-label'>Embedding model</span><br><strong>{escape(EMBEDDING_MODEL)}</strong>", unsafe_allow_html=True)
        st.markdown(
            f"<br><span class='sidebar-label'>Course library</span><br>"
            f"<span class='status-pill'>Connected · ChromaDB ({total_chunks} chunks)</span>",
            unsafe_allow_html=True,
        )

    # Header & course strip
    mark, banner_title, banner_detail = _course_banner_info(selected_course)
    st.markdown(
        f"""
        <div class="app-heading">
          <h1>Course Q&amp;A</h1>
          <p>Ask about your course materials in English or Arabic · Mode: <strong>{escape(selected_mode)}</strong></p>
        </div>
        <div class="course-strip">
          <div class="course-mark">{escape(mark)}</div>
          <div>
            <div class="course-name">{escape(banner_title)}</div>
            <div class="course-detail">{escape(banner_detail)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    messages = active_chat["messages"]
    if not messages:
        st.markdown(
            """
            <div class="empty-state">
              <div class="empty-icon">\U0001F4AC</div>
              <h3>Start with a course question</h3>
              <p>Ask about a concept, lecture, or assignment in Arabic or English. Answers stay strictly grounded in your indexed course materials.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<div class='section-heading'>Your conversation</div>", unsafe_allow_html=True)

    for message in messages:
        with st.chat_message(message["role"]):
            direction = text_direction(message["content"])
            st.markdown(_format_message_html(message["content"], direction), unsafe_allow_html=True)
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))
                render_reasoning(message.get("reasoning"), message.get("retrieval_trace"))
                timings = message.get("timings")
                if timings:
                    render_metrics(timings)

    question = st.chat_input("Ask about course materials (English or Arabic)...")

    if question:
        if not messages:
            active_chat["title"] = (question[:42] + "...") if len(question) > 42 else question
        messages.append({"role": "user", "content": question})
        active_chat["updated_at"] = _now_timestamp()
        save_chat_store(store)

        with st.chat_message("user"):
            direction = text_direction(question)
            st.markdown(_format_message_html(question, direction), unsafe_allow_html=True)

        with st.chat_message("assistant"):
            spinner_text = {
                "Standard RAG": "Searching course materials and generating answer...",
                "Corrective RAG": "Searching, grading evidence, and generating answer...",
                "Agentic RAG": "Planning steps, verifying passages, running tools, and generating answer...",
            }[selected_mode]
            with st.spinner(spinner_text):
                try:
                    answer, sources, reasoning, trace, timings = run_rag_pipeline(
                        question,
                        selected_course,
                        selected_mode,
                    )

                    answer_direction = text_direction(answer)
                    st.markdown(_format_message_html(answer, answer_direction), unsafe_allow_html=True)

                    render_sources(sources)
                    render_reasoning(reasoning, trace)
                    render_metrics(timings)

                    messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "question": question,
                            "sources": sources,
                            "reasoning": reasoning,
                            "retrieval_trace": trace,
                            "timings": timings,
                        }
                    )
                    active_chat["updated_at"] = _now_timestamp()
                    save_chat_store(store)

                except Exception as error:
                    st.error(
                        "The request could not be completed. Check that Ollama is running "
                        "and the configured models are available."
                    )
                    st.exception(error)


if __name__ == "__main__":
    main()
