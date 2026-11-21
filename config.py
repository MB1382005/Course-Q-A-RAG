
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Data directories
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Ollama models
LLM_MODEL = "qwen3:8b"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"

# RAG settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
    