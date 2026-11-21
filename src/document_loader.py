
from pathlib import Path

import pymupdf
import pandas as pd
from docx import Document


def _original_filename(file_path):
    """Strip the OCW download hash prefix while keeping the real filename."""
    name = Path(file_path).name
    if len(name) > 33 and name[32] == "_" and all(c in "0123456789abcdef" for c in name[:32].lower()):
        return name[33:]
    return name


# ==========================================
# 1. PDF Loader
# ==========================================

def load_pdf(file_path, course_name):
    documents = []

    with pymupdf.open(file_path) as pdf:

        for page_number, page in enumerate(pdf, start=1):

            text = page.get_text("text").strip()

            if text:
                documents.append({
                    "text": text,
                    "metadata": {
                        "course": course_name,
                        "source": _original_filename(file_path),
                        "file_type": "pdf",
                        "page": page_number
                    }
                })

    return documents


# ==========================================
# 2. DOCX Loader
# ==========================================

def load_docx(file_path, course_name):

    doc = Document(file_path)

    text = "\n".join(
        paragraph.text
        for paragraph in doc.paragraphs
        if paragraph.text.strip()
    )

    if not text:
        return []

    return [{
        "text": text,
        "metadata": {
            "course": course_name,
            "source": Path(file_path).name,
            "file_type": "docx"
        }
    }]


# ==========================================
# 3. TXT Loader
# ==========================================

def load_txt(file_path, course_name):

    text = Path(file_path).read_text(
        encoding="utf-8",
        errors="replace"
    ).strip()

    if not text:
        return []

    return [{
        "text": text,
        "metadata": {
            "course": course_name,
            "source": Path(file_path).name,
            "file_type": "txt"
        }
    }]


# ==========================================
# 4. CSV Loader
# ==========================================

def load_csv(file_path, course_name):

    dataframe = pd.read_csv(file_path).fillna("")

    documents = []

    for index, row in dataframe.iterrows():

        row_text = " | ".join(
            f"{column}: {value}"
            for column, value in row.items()
        )

        if row_text.strip():

            documents.append({
                "text": row_text,
                "metadata": {
                    "course": course_name,
                    "source": _original_filename(file_path),
                    "file_type": "csv",
                    "row": int(index) + 1
                }
            })

    return documents


# ==========================================
# 5. Main Document Loader
# ==========================================

def load_document(file_path, course_name):

    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        return load_pdf(file_path, course_name)

    elif extension == ".docx":
        return load_docx(file_path, course_name)

    elif extension == ".txt":
        return load_txt(file_path, course_name)

    elif extension == ".csv":
        return load_csv(file_path, course_name)

    else:
        return []


# ==========================================
# 6. Load Entire Course Folder
# ==========================================

def load_course_folder(folder_path, course_name):
    """Load all supported documents recursively from a course folder."""

    folder = Path(folder_path)

    supported_extensions = {".pdf", ".docx", ".txt", ".csv"}

    all_documents = []

    if not folder.is_dir():
        raise NotADirectoryError(f"Course folder does not exist: {folder}")

    for file_path in sorted(folder.rglob("*")):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in supported_extensions:
            continue

        try:
            documents = load_document(file_path, course_name)
            all_documents.extend(documents)

            print(f"Loaded: {file_path.name} ({len(documents)} documents)")

        except Exception as error:
            print(f"Could not load {file_path.name}: {error}")

    return all_documents
