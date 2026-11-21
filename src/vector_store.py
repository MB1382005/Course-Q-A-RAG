
import chromadb
import ollama

from config import EMBEDDING_MODEL, CHROMA_DIR


# 1. Create persistent ChromaDB client
client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)


# 2. Create or access collection
collection = client.get_or_create_collection(
    name="course_materials"
)


# 3. Store chunks in batches
def store_chunks(chunks, batch_size=16):

    if not chunks:
        print("No chunks to store.")
        return
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    total_chunks = len(chunks)

    print("Total chunks:", total_chunks)
    print("Batch size:", batch_size)

    # Process chunks batch by batch
    for start in range(0, total_chunks, batch_size):

        end = min(start + batch_size, total_chunks)

        batch = chunks[start:end]

        # Extract texts
        texts = [
            chunk["text"]
            for chunk in batch
        ]

        # Extract metadata
        metadatas = [
            chunk["metadata"]
            for chunk in batch
        ]

        # Stable IDs prevent collisions across courses and repeated ingestion.
        ids = []
        for chunk in batch:
            metadata = chunk["metadata"]
            course = metadata.get("course", "")
            source = metadata.get("source", "")
            page = metadata.get("page", metadata.get("row", ""))
            chunk_id = metadata.get("chunk_id", "")
            identity = f"{course}|{source}|{page}|{chunk_id}|{chunk['text']}"
            import hashlib
            ids.append(hashlib.sha256(identity.encode("utf-8")).hexdigest())

        print(f"\nProcessing batch: {start + 1} - {end}")

        # Generate embeddings
        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=texts
        )

        embeddings = response.get("embeddings", [])

        if len(embeddings) != len(batch):
            raise ValueError("Embedding count does not match input chunk count")

        # Validate embedding dimensions
        dimensions = {len(vector) for vector in embeddings}
        if len(dimensions) != 1:
            raise ValueError("Embedding vectors have inconsistent dimensions")
        existing_dimension = collection.peek().get("embeddings")
        if len(existing_dimension) > 0 and len(existing_dimension[0]) != next(iter(dimensions)):
            raise ValueError("Embedding dimension does not match existing collection")

        # Store batch in ChromaDB
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(f"Stored successfully: {len(batch)} chunks")

    print("\n========== STORAGE COMPLETED ==========")

    print("Total stored:", collection.count())
    print("Vector dimensions:", 1024)
