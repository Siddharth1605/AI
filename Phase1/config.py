from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DOCUMENT_DIRECTORY = BASE_DIR / "documents"
CHROMA_DIRECTORY = BASE_DIR / "chroma_db"

COLLECTION_NAME = "engineering_knowledge"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "gemma3"

# Chunking configuration
MAX_CHUNK_CHARACTERS = 2500
CHUNK_OVERLAP_BLOCKS = 1

# Retrieval configuration
VECTOR_CANDIDATES = 20
BM25_CANDIDATES = 20
FUSED_CANDIDATES = 12
RERANK_CANDIDATES = 8
FINAL_CONTEXT_CHUNKS = 4

# Reciprocal Rank Fusion configuration
RRF_CONSTANT = 60
VECTOR_WEIGHT = 0.55
BM25_WEIGHT = 0.45
