import os

# === Paths ===
EMB_DIR = "embeddings"
DATA_DIR = "data_fh_wedel"
META_FILE = "fhwedel_docs.json"
EMB_CACHE_FILE = os.path.join(EMB_DIR, "emb_cache.json")

# === Embedding ===
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-qwen3-embedding-0.6b")
CHUNK_SIZE = 200
CHUNK_OVERLAP = 30
EMB_BATCH = 500 

# === Qdrant ===
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "pdf_chunks"
UPLOAD_BATCH = 800


# === Chat model ===
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen/qwen3-4b-2507")
CLASSIFY_MODEL = os.getenv("CLASSIFIE_MODEL", "qwen/qwen3-4b-2507")

# === OpenAI client (can point to LM Studio or OpenAI) ===
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "not-needed-for-local")

