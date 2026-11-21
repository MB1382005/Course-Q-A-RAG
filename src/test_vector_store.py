
from pathlib import Path

from src.document_loader import load_course_folder
from src.chunking import create_chunks
from src.vector_store import store_chunks, collection


BASE_DIR = Path(__file__).resolve().parent.parent

COURSE_FOLDER = (
    BASE_DIR
    / "data"
    / "course_1"
    / "static_resources"
)


# Load documents
documents = load_course_folder(
    COURSE_FOLDER,
    "Database Management Systems"
)

# Create chunks
chunks = create_chunks(documents)

print("Documents:", len(documents))
print("Chunks:", len(chunks))


# Store chunks and embeddings
store_chunks(chunks)


# Verify database
print("\n========== DATABASE SUMMARY ==========")

print("Stored records:", collection.count())
