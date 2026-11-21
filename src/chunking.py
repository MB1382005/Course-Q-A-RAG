
from config import CHUNK_SIZE, CHUNK_OVERLAP


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):

    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")
    if overlap < 0:
        raise ValueError("Overlap cannot be negative.")
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    chunks = []
    text = text.strip()
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Prefer a nearby paragraph/sentence/word boundary, but retain the
        # original fixed-size behavior when no useful boundary is available.
        if end < len(text):
            lower_bound = start + max(1, chunk_size // 2)
            candidates = [(text.rfind(token, lower_bound, end), token)
                          for token in ("\n\n", ". ", "? ", "! ", " ")]
            boundary, separator = max(candidates, key=lambda item: item[0])
            if boundary >= lower_bound:
                end = boundary + len(separator)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
        if start < end and not text[start - 1].isspace() and not text[start].isspace():
            next_space = text.find(" ", start, end)
            if next_space >= 0:
                start = next_space + 1

    return chunks


def create_chunks(documents):

    all_chunks = []

    for document in documents:

        text = document["text"]
        metadata = document["metadata"]

        chunks = split_text(text)

        for index, chunk in enumerate(chunks):

            all_chunks.append({
                "text": chunk,
                "metadata": {
                    **metadata,
                    "chunk_id": index
                }
            })

    return all_chunks
