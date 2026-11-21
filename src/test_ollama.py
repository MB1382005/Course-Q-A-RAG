
import sys
from pathlib import Path

# Add project root to Python search path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import ollama
from config import LLM_MODEL, EMBEDDING_MODEL


print("Testing LLM Model...")

response = ollama.chat(
    model=LLM_MODEL,
    messages=[
        {
            "role": "user",
            "content": "Explain RAG in two simple sentences."
        }
    ]
)

print("\nLLM Response:")
print(response["message"]["content"])


print("\nTesting Embedding Model...")

embedding_response = ollama.embed(
    model=EMBEDDING_MODEL,
    input="Retrieval Augmented Generation"
)

vector = embedding_response["embeddings"][0]

print("\nEmbedding Test Successful!")
print("Vector dimensions:", len(vector))
