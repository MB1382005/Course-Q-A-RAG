# Course Q&A RAG (Standard + Corrective + Agentic RAG)

Streamlit course assistant with a course selector for three independent course collections in the shared ChromaDB collection:

- Database Management Systems (MIT OpenCourseWare 6.830, Fall 2010)
- Human-Computer Interaction (Course 2)
- Computer Security (CS 401, Course 3)

New chats default to **General - all courses**, which searches all three course collections in one query. You can still choose a single course from the sidebar when you want a narrower answer.

## Run

From PowerShell in this directory:

```powershell
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

Ollama must be running locally with `qwen3:8b` and `qwen3-embedding:0.6b` available. The application reads the existing `chroma_db` collection named `course_materials` and does not ingest or rebuild it at startup. Retrieval filters by the selected course metadata, so chunks from other courses are not mixed into the answer.

Chat sessions are stored locally in `chat_history.json`, so conversations and the selected course remain available after a browser refresh. Each assistant answer also includes a short evidence summary with a retrieved course excerpt; this is a concise explanation of its support, not private chain-of-thought.

## Corrective RAG flow

For each question, the app retrieves candidate chunks and grades them with `qwen3:8b` for direct relevance. If no chunk passes, it asks the same model to rewrite the search query once, retrieves again, and grades the new candidates. Only graded chunks are passed to answer generation or shown as answer sources. If neither retrieval attempt yields accepted evidence, the app returns the existing insufficient-materials response (or a course handoff when another course is a better match). The “Why this answer?” panel notes when query correction was used.

The grader and rewrite calls use Ollama too, so the configured model must remain available. If grading fails or returns malformed output, a conservative lexical/strong-distance fallback is used; when that fallback accepts no evidence, generation is skipped.

## Standard RAG mode

Select **Standard RAG** in the sidebar to run the original direct pipeline: retrieve the nearest course chunks, then generate an answer from those passages. It does not grade chunks or rewrite/retry the query. All three modes share the same course selector, source attribution, and grounded answer generation.

## Agentic RAG mode

Select **Agentic RAG** in the sidebar's **RAG mode** menu. The planner turns a question into up to three retrieval queries and may select evidence summarization, comparison, or arithmetic calculation. Each retrieval query passes through the corrective grader/rewrite flow; only accepted passages are given to final answer generation and shown as sources. The interface exposes the plan summary and tool outputs. If the planner is unavailable or returns invalid JSON, the agent falls back to retrieving the original question. The calculator evaluates only arithmetic expressions written in the question; it does not execute arbitrary code. The assistant still answers from verified course passages only.

## Validation

Run the isolated regression checks with:

```powershell
.\venv\Scripts\python.exe -m unittest src.test_quality -v
```

The older `src/test_loader.py`, `src/test_chunking.py`, and `src/test_vector_store.py` files are manual scripts; notably, `test_vector_store.py` writes to the live collection. Do not run it as a routine test unless ingestion is intended.

## Add newly supplied course materials

Put supported files (`.pdf`, `.docx`, `.txt`, or `.csv`) in that course's folder under `data/`, then add or update its entry in `src/course_catalog.py`. To ingest a course into the existing collection, run:

```powershell
.\venv\Scripts\python.exe -m src.index_course "Human-Computer Interaction"
```

The indexer uses stable IDs and upserts into the existing collection. It does not delete or recreate the collection. Avoid re-indexing Course 1 unless its source materials have changed.

You can also upload a `.zip` from the sidebar's **Upload course ZIP** section. Choose the destination course and click **Extract and index ZIP**. The app creates a folder named after the archive under that course's `uploaded_archives/` directory, extracts supported PDF, DOCX, TXT, and CSV files, and indexes them with the selected course metadata. ZIP extraction rejects oversized archives and skips unsafe paths, symlinks, and unsupported file types.

## Indexing notes

PDF extraction stores one-based PDF page numbers. Newly ingested OCW filenames omit the leading 32-character hexadecimal download hash. Existing ChromaDB metadata is preserved as-is; reindexing is not performed automatically. New vector IDs are stable hashes of course, source, page/row, chunk ID, and text, preventing collisions across batches and courses.

Chunking prefers sentence, paragraph, or word boundaries within the existing 800-character size and 150-character overlap settings. Rebuilding existing vectors would be required for those chunking changes to affect current search results.
