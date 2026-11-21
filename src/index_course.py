"""Add newly supplied course files to the existing Chroma collection."""

import argparse
from pathlib import Path

from src.chunking import create_chunks
from src.course_catalog import COURSES
from src.document_loader import load_course_folder
from src.vector_store import store_chunks


def index_course(folder: Path, course_name: str):
    if not folder.is_dir():
        raise NotADirectoryError(f"Course folder does not exist: {folder}")

    documents = load_course_folder(folder, course_name)
    chunks = create_chunks(documents)
    if not chunks:
        print(f"No supported readable documents found in {folder}.")
        return 0

    store_chunks(chunks)
    return len(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("course", choices=COURSES)
    args = parser.parse_args()
    course = COURSES[args.course]
    if not course["folder"].is_dir():
        raise NotADirectoryError(course["folder"])
    count = index_course(course["folder"], course["name"])
    print(f"Added {count} chunks for {course['name']}.")


if __name__ == "__main__":
    main()
