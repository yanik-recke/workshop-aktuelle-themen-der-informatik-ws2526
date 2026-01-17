import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

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
EMBEDDING_DIM = 1024  # Qwen3-0.6b embedding model produces 1024-dim vectors

# === Modelle (über OpenAI-kompatibles API, z.B. LM Studio) ===
#OPENAI_CHAT_MODEL = "deepseek-r1:1.5b"
OPENAI_EMBED_MODEL = "qwen3-embedding:0.6b"
OPENAI_CHAT_MODEL = "gpt-4o"
#OPENAI_EMBED_MODEL = "text-embedding-qwen3-embedding-0.6b"

OPENAI_CLASSIFIER_MODEL = OPENAI_CHAT_MODEL  # eigenes Modell möglich

# === API Base URLs ===
# Separate URLs for embeddings (local Ollama) and chat (OpenAI API)
OPENAI_EMBED_BASE_URL = os.getenv("OPENAI_EMBED_BASE_URL", "http://localhost:11434/v1")  # Ollama
OPENAI_CHAT_BASE_URL = os.getenv("OPENAI_CHAT_BASE_URL", "https://api.openai.com/v1")    # OpenAI API

# WICHTIG: Setze diese Umgebungsvariablen:
#   export OPENAI_API_KEY="sk-..."  # Your OpenAI API key
#   export OPENAI_EMBED_BASE_URL="http://localhost:11434/v1"  # Ollama for embeddings
#   export OPENAI_CHAT_BASE_URL="https://api.openai.com/v1"   # OpenAI for chat

# === RAG / Chat ===
TOP_K = 5  # Increase for more diverse results
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
