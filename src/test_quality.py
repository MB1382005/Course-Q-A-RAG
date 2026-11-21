"""Fast, isolated regression checks for the RAG pipeline helpers."""
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from io import BytesIO
from zipfile import ZipFile

from src.chunking import split_text, create_chunks
from src.generator import generate_answer
from src.retriever import retrieve_documents
from src.document_loader import load_pdf
from src.corrective_rag import corrective_retrieve
from src.agentic_rag import run_agent, _safe_calculate
from src.zip_ingestion import extract_course_zip


class ChunkingTests(unittest.TestCase):
    def test_sentence_boundary_and_overlap_preserve_text(self):
        text = "First sentence. Second sentence. Third sentence."
        chunks = split_text(text, chunk_size=24, overlap=5)
        self.assertTrue(all(chunk.strip() for chunk in chunks))
        self.assertEqual("First sentence. Second sentence. Third sentence.",
                         " ".join(chunks).replace("  ", " "))

    def test_chunk_metadata_is_preserved(self):
        result = create_chunks([{"text": "A" * 40,
                                 "metadata": {"course": "C1", "source": "a.pdf", "page": 2}}])
        self.assertEqual("a.pdf", result[0]["metadata"]["source"])
        self.assertEqual(2, result[0]["metadata"]["page"])
        self.assertEqual(0, result[0]["metadata"]["chunk_id"])

    def test_invalid_sizes_fail(self):
        with self.assertRaises(ValueError):
            split_text("abc", chunk_size=0)


class LoaderTests(unittest.TestCase):
    @patch("src.document_loader.pymupdf.open")
    def test_pdf_uses_original_filename_and_one_based_page(self, open_pdf):
        from contextlib import nullcontext
        page = type("Page", (), {"get_text": lambda self, mode: "page text"})()
        pdf = type("PDF", (), {"__iter__": lambda self: iter([page])})()
        open_pdf.return_value = nullcontext(pdf)
        result = load_pdf("D:/docs/0123456789abcdef0123456789abcdef_MIT6_830F10_lec01.pdf", "Course 1")
        self.assertEqual("MIT6_830F10_lec01.pdf", result[0]["metadata"]["source"])
        self.assertEqual(1, result[0]["metadata"]["page"])


class GenerationTests(unittest.TestCase):
    def test_empty_context_refuses_in_question_language(self):
        self.assertIn("\u0644\u0627 \u062a\u062a\u0648\u0641\u0631", generate_answer("\u0645\u0627 \u0647\u0648\u061f", []))
        self.assertIn("not provide enough", generate_answer("What is it?", []))

    def test_source_collection_uses_retriever_metadata_and_handles_raw_chroma(self):
        from app import collect_sources
        retrieved = [
            {"text": "a", "metadata": {"source": "MIT6_830F10_lec01.pdf", "page": 1}},
            {"text": "b", "metadata": {"source": "MIT6_830F10_lec01.pdf", "page": 1}},
            {"text": "c", "metadata": {"source": "MIT6_830F10_lec01.pdf", "page": 2}},
            {"text": "d", "metadata": {"source": "no-page.pdf"}},
            {"text": "e", "metadata": {}},
        ]
        self.assertEqual([
            {"file": "MIT6_830F10_lec01.pdf", "page": 1},
            {"file": "MIT6_830F10_lec01.pdf", "page": 2},
            {"file": "no-page.pdf", "page": None},
        ], collect_sources(retrieved))
        self.assertEqual([{"file": "raw.pdf", "page": 3}],
                         collect_sources([{"source": "raw.pdf", "page": 3}]))

    @patch("src.generator.ollama.chat")
    def test_blank_model_output_fails_closed(self, chat):
        chat.return_value = {"message": {"content": "  "}}
        docs = [{"text": "Relevant course passage.", "metadata": {"source": "a.pdf", "page": 1}}]
        self.assertIn("not provide enough", generate_answer("What is unsupported?", docs))

    @patch("src.generator.ollama.chat")
    def test_answer_uses_course_only_prompt_and_keep_alive(self, chat):
        chat.return_value = {"message": {"content": "Supported response."}}
        docs = [{"text": "A DBMS manages databases.",
                 "metadata": {"source": "lecture.pdf", "page": 1}}]
        self.assertEqual("Supported response.", generate_answer("What is a DBMS?", docs))
        kwargs = chat.call_args.kwargs
        self.assertEqual("30m", kwargs["keep_alive"])
        self.assertFalse(kwargs["think"])
        self.assertIn("ONLY facts explicitly supported", kwargs["messages"][0]["content"])
        prompt = kwargs["messages"][0]["content"]
        self.assertIn("Modern Standard Arabic", prompt)
        self.assertIn("يتوسط بين التطبيقات والبيانات", prompt)


class RetrievalTests(unittest.TestCase):
    @patch("src.retriever.ollama.embed")
    @patch("src.retriever.collection.query")
    def test_course_filter_and_duplicate_removal(self, query, embed):
        embed.return_value = {"embeddings": [[0.0] * 1024]}
        query.return_value = {
            "documents": [["same relevant course passage", "same relevant course passage"]],
            "metadatas": [[{"source": "a.pdf", "page": 1, "course": "Database Management Systems"}] * 2],
            "distances": [[0.1, 0.2]],
        }
        results = retrieve_documents("  question   normalized ")
        self.assertEqual(1, len(results))
        self.assertEqual("Database Management Systems", query.call_args.kwargs["where"]["course"])
        self.assertEqual("question normalized", embed.call_args.kwargs["input"])


class CorrectiveRetrievalTests(unittest.TestCase):
    def test_rewrites_and_retrieves_when_initial_evidence_is_rejected(self):
        calls = []
        def retrieve(query, course_name):
            calls.append(query)
            return [{"text": query, "metadata": {"source": "course.pdf", "page": 1}}]
        result = corrective_retrieve(
            "What is X?", "Course",
            retriever=retrieve,
            grader=lambda query, docs: docs if query == "Explain X in the lecture" else [],
            rewriter=lambda query: "Explain X in the lecture",
        )
        self.assertEqual(["What is X?", "Explain X in the lecture"], calls)
        self.assertTrue(result["corrected"])
        self.assertTrue(result["sufficient"])
        self.assertEqual(1, len(result["documents"]))

    def test_does_not_rewrite_when_evidence_passes(self):
        result = corrective_retrieve(
            "Question", "Course",
            retriever=lambda query, course_name: [{"text": "passage"}],
            grader=lambda query, docs: docs,
            rewriter=lambda query: self.fail("query must not be rewritten"),
        )
        self.assertFalse(result["corrected"])
        self.assertTrue(result["sufficient"])

    def test_returns_no_accepted_evidence_after_second_failure(self):
        result = corrective_retrieve(
            "Question", "Course",
            retriever=lambda query, course_name: [{"text": "irrelevant"}],
            grader=lambda query, docs: [],
            rewriter=lambda query: "rewritten question",
        )
        self.assertFalse(result["sufficient"])
        self.assertEqual([], result["documents"])


class AgenticRagTests(unittest.TestCase):
    def test_agent_executes_multiple_search_steps_and_deduplicates_evidence(self):
        plan = {"steps": [
            {"tool": "retrieve", "query": "Concept A", "purpose": "Find A"},
            {"tool": "retrieve", "query": "Concept B", "purpose": "Find B"},
            {"tool": "compare", "purpose": "Compare them"},
        ], "answer_strategy": "Compare supported details", "planner": "test"}
        def retrieve(query, course_name):
            return [{"text": f"Evidence for {query}", "metadata": {"source": "lecture.pdf", "page": 1}}]
        result = run_agent("Compare A and B", "Course", retriever_fn=retrieve,
                           grader_fn=lambda question, docs: docs,
                           planner_fn=lambda question, course: plan)
        self.assertEqual(2, len(result["documents"]))
        self.assertEqual(2, len(result["attempts"]))
        self.assertEqual("compare", result["tool_outputs"][0]["tool"])

    def test_calculator_only_evaluates_arithmetic_expression(self):
        self.assertEqual([{"expression": "12 / 3 + 2", "result": 6.0}],
                         _safe_calculate("Calculate 12 / 3 + 2"))
        self.assertEqual([], _safe_calculate("Calculate __import__('os').system('whoami')"))

    def test_agent_plan_fallback_includes_retrieval(self):
        from src.agentic_rag import plan_question
        with patch("src.agentic_rag.ollama.chat", side_effect=RuntimeError("offline")):
            plan = plan_question("What is X?", "Course")
        self.assertEqual("fallback", plan["planner"])
        self.assertEqual("What is X?", plan["steps"][0]["query"])


class StandardRagTests(unittest.TestCase):
    def test_direct_retrieval_keeps_course_filter_and_skips_grading(self):
        from src.retriever import retrieve_documents as direct_retrieve
        with patch("src.retriever.ollama.embed", return_value={"embeddings": [[0.0] * 1024]}), \
             patch("src.retriever.collection.query", return_value={
                 "documents": [["A directly retrieved course passage with enough text."]],
                 "metadatas": [[{"source": "a.pdf", "page": 1, "course": "Course A"}]],
                 "distances": [[0.1]],
             }) as query:
            docs = direct_retrieve("question", course_name="Course A")
        self.assertEqual(1, len(docs))
        self.assertEqual({"course": "Course A"}, query.call_args.kwargs["where"])
        self.assertNotIn("relevance", docs[0])


class ZipIngestionTests(unittest.TestCase):
    def test_extracts_supported_files_to_named_folder_and_skips_traversal(self):
        archive_bytes = BytesIO()
        with ZipFile(archive_bytes, "w") as archive:
            archive.writestr("lectures/week1.txt", "Course lecture content")
            archive.writestr("../../outside.txt", "should not escape")
            archive.writestr("notes.exe", "unsupported")
        with TemporaryDirectory() as temp_dir:
            destination, count = extract_course_zip(
                archive_bytes.getvalue(), "Course Pack.zip", Path(temp_dir) / "course"
            )
            self.assertEqual("Course Pack", destination.name)
            self.assertEqual(1, count)
            self.assertEqual("Course lecture content", (destination / "lectures" / "week1.txt").read_text())
            self.assertFalse((Path(temp_dir) / "outside.txt").exists())

    def test_rejects_zip_without_supported_files(self):
        archive_bytes = BytesIO()
        with ZipFile(archive_bytes, "w") as archive:
            archive.writestr("readme.md", "not indexed")
        with TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "no supported"):
                extract_course_zip(archive_bytes.getvalue(), "empty.zip", temp_dir)


if __name__ == "__main__":
    unittest.main()
