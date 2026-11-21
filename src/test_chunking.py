
from pathlib import Path

from src.document_loader import load_course_folder
from src.chunking import create_chunks


BASE_DIR = Path(__file__).resolve().parent.parent

COURSE_FOLDER = BASE_DIR / "data" / "course_1" / "static_resources"


documents = load_course_folder(
    COURSE_FOLDER,
    "Database Management Systems"
)

chunks = create_chunks(documents)


print("\n========== CHUNKING SUMMARY ==========")

print("Total documents:", len(documents))
print("Total chunks:", len(chunks))


if chunks:

    print("\n========== FIRST CHUNK ==========")

    print(chunks[0]["text"])

    print("\n========== METADATA ==========")

    print(chunks[0]["metadata"])
