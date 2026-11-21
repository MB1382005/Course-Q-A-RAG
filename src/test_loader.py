
from pathlib import Path

from src.document_loader import load_course_folder


# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent


# Course materials directory
COURSE_FOLDER = (
    BASE_DIR
    / "data"
    / "course_1"
    / "static_resources"
)


# Load course materials
documents = load_course_folder(
    COURSE_FOLDER,
    "Database Management Systems"
)


# Display results
print("\n========== LOADING SUMMARY ==========")

print("Total extracted documents:", len(documents))


# Display first document
if documents:

    print("\n========== FIRST DOCUMENT ==========")

    print(documents[0]["text"][:500])

    print("\n========== METADATA ==========")

    print(documents[0]["metadata"])

else:

    print("No documents were extracted.")
