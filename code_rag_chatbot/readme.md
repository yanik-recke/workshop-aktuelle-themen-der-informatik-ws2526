# FH Wedel RAG Chatbot

A local retrieval-augmented chatbot for answering questions about FH Wedel study programs using official PDFs (module handbooks, study regulations, etc.).

---

## Workflow

### 1 Install dependencies
```bash
pip install -r requirements.txt
```

---

### 2️ Download PDFs
Fetch all study documents into the local data folder:
```bash
python download_pdfs.py
```

---

### 3️ Start Qdrant (vector database)
```bash
docker run -p 6333:6333 -d qdrant/qdrant
```

Qdrant runs at [http://localhost:6333](http://localhost:6333).

---

### 4️ Ingest documents
Extract, embed, and store chunks in Qdrant:
```bash
python ingest.py
```

---

### 5️ Start the chat interface
```bash
python chat_cli.py
```



##  Model Configuration
Edit `config.py` or set environment variables before running:

```bash
export OPENAI_BASE_URL="http://localhost:1234/v1"
export OPENAI_API_KEY="not-needed"
export EMBED_MODEL="text-embedding-qwen3-embedding-0.6b"
export CHAT_MODEL="qwen/qwen3-4b-2507"
```

---

**Summary:**  
Install → Download PDFs → Start Qdrant → Ingest → Chat.
