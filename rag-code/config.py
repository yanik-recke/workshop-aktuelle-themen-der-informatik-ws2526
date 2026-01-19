import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file (override=True to replace system env vars with .env values)
load_dotenv(override=True)

# === Pfade ===
PROJECT_ROOT = Path(__file__).parent
# Ordner mit Markdown-Dateien
DATA_DIR = PROJECT_ROOT / "data" / "documents" / "md-docs"
META_FILE = PROJECT_ROOT / "data" / "documents" / "meta.json"

# === Qdrant ===
# Use env, fallback to localhost as default value
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = "fh_wedel_pdfs"

# Achtung: Muss zur Dimension deines Embedding-Modells passen
# (z.B. 1536 wie bei vielen OpenAI-kompatiblen Embedding-Modellen)
EMBEDDING_DIM = 1536  # OpenAI text-embedding-3-small produces 1536-dim vectors

# === Modelle ===
# OpenAI
OPENAI_CHAT_MODEL = "gpt-4o-mini"  # Fast and capable for chat
OPENAI_EMBED_MODEL = "text-embedding-3-small"  # OpenAI embedding model (1536 dim)
OPENAI_RERANK_MODEL = "gpt-4o-mini"  # For reranking chunks

# For local LM Studio (set OPENAI_BASE_URL in .env)
# OPENAI_CHAT_MODEL = "gpt-oss-20b"
# OPENAI_EMBED_MODEL = "qwen3-embedding:0.6b"
# OPENAI_RERANK_MODEL = "gpt-oss-20b"

OPENAI_CLASSIFIER_MODEL = OPENAI_CHAT_MODEL

# WICHTIG: Setze diese Umgebungsvariablen für LM Studio o.ä.:
#   export OPENAI_API_KEY="dummy"
#   export OPENAI_BASE_URL="http://localhost:1234/v1"

# === RAG / Chat ===
TOP_K = 10  # Increase for more diverse results
USE_LLM_RERANK = True  # Set True to enable LLM-based reranking (slower but better)
MAX_MEMORY_TURNS = 5
MAX_CONTEXT_TOKENS = 5000
# === Fachhochschule Wedel Kategorien ===
DEGREES = ["Bachelor", "Master"]

PROGRAMS = [
    "Informatik",
    "Wirtschaftsinformatik",
    "Wirtschaftsingenieurwesen",
    "Betriebswirtschaftslehre",
    "E-Commerce",
    "Data Science & Artificial Intelligence",
    "IT-Sicherheit",
    "Sustainable & Digital Business Management",
    "Computer Games Technology",
    "Smart Technology",
    "IT-Management, Consulting (und Auditing)",
]

DOCTYPES = {
    "Modulhandbuch": "Modulhandbuch mit genaueren Inhalten / Lernzielen aller Module des Studiengangs",
    "Studien- und Prüfungsordnung": "Voraussetzungen und Regeln innerhalb des Studiums",
    "Modulübersicht": "Übersicht der Module über den Verlauf des Studiums",
    "Studienverlaufsplan": "Tabellarischer Verlaufsplan des Studiengangs (Welche Module finden wann statt und wie sind diese gewichtet)",
}

STATUSES = ["aktuell", "archiviert"]
