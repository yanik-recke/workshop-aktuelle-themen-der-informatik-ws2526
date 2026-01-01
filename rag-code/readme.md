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


### 4️ Ingest documents
Extract, embed, and store chunks in Qdrant:
```bash
python indexing_pipeline.py
```

---

### 5️ Start the chat interface
```bash
python chat_cli.py
```



##  Model Configuration
Edit Models in `config.py` and set environment variables before running:

```bash
export OPENAI_BASE_URL="http://localhost:1234/v1"
export OPENAI_API_KEY="not-needed"
```

### Windows
```bash
set OPENAI_API_KEY=dummy
set OPENAI_BASE_URL=http://localhost:1234/v1
```

---

**Summary:**  
Install → Download PDFs → Start Qdrant → Ingest → Chat.
