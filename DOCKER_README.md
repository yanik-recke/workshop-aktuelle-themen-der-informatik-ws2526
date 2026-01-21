# Docker Setup for FH Wedel RAG Chatbot

## Quick Start

### 1. Clone and Navigate
```bash
cd /path/to/workshop-aktuelle-themen-der-informatik-ws2526
```

### 2. Environment Configuration
Copy the example environment file and adjust as needed:
```bash
cp .env.example .env
```

### 3. Start Services
Start Qdrant and Ollama:
```bash
docker-compose up -d qdrant ollama
```

Wait for services to be ready (~30 seconds).

### 4. Download Required Models (Ollama)
If using Ollama, download the required models:
```bash
# Download chat model (Qwen 2.5 7B - ~4.7GB)
docker exec -it fh-wedel-ollama ollama pull qwen2.5:7b-instruct-q8_0

# Download embedding model (Qwen3 0.6B - ~600MB)
docker exec -it fh-wedel-ollama ollama pull nomic-embed-text
```

**Note:** The exact model names in `config.py` may need adjustment for Ollama compatibility. See [Model Configuration](#model-configuration) below.

### 5. Download FH Wedel Documents
Download PDFs from FH Wedel website:
```bash
docker-compose --profile setup run --rm download-docs
```

This will populate `rag-code/data_fh_wedel/` with study documents.

### 6. Index Documents
Process and embed documents into Qdrant:
```bash
docker-compose --profile setup run --rm indexing
```

This step may take 10-30 minutes depending on the number of documents and your hardware.

### 7. Start the Chat Interface
```bash
docker-compose up rag-app
```

Or run interactively:
```bash
docker-compose run --rm rag-app
```

## Architecture

The Docker setup consists of three main services:

```
┌─────────────────┐
│   rag-app       │  Python 3.13 Application
│   (port: none) │  - RAG pipeline
│                 │  - CLI interface
└────────┬────────┘
         │
         ├──────────────┐
         │              │
┌────────▼────────┐ ┌──▼──────────────┐
│    qdrant       │ │    ollama       │
│  (port: 6333)   │ │  (port: 11434)  │
│  Vector DB      │ │  LLM Service    │
└─────────────────┘ └─────────────────┘
```

### Services

1. **qdrant** - Vector database for document embeddings
   - Port: 6333 (HTTP API), 6334 (gRPC)
   - Data persisted in Docker volume `qdrant_data`

2. **ollama** - Local LLM inference server
   - Port: 11434
   - Models stored in Docker volume `ollama_data`
   - Alternative to LM Studio

3. **rag-app** - Main RAG chatbot application
   - Runs `chat_cli.py` by default
   - Connects to qdrant and ollama
   - Interactive terminal interface

## Model Configuration

### Using Ollama (Default)

The docker-compose.yml is configured for Ollama by default. However, you need to adjust model names in `rag-code/config.py`:

```python
# Edit rag-code/config.py
OPENAI_CHAT_MODEL = "qwen2.5:7b-instruct-q8_0"
OPENAI_EMBED_MODEL = "nomic-embed-text"
```

Available Ollama models:
- Chat: `qwen2.5:7b-instruct-q8_0`, `mistral`, `llama2`, `neural-chat`
- Embeddings: `nomic-embed-text`, `all-minilm`

List downloaded models:
```bash
docker exec -it fh-wedel-ollama ollama list
```

### Using LM Studio (External)

If you prefer LM Studio running on your host machine:

1. Start LM Studio and load models
2. Update `.env`:
   ```bash
   OPENAI_BASE_URL=http://host.docker.internal:1234/v1
   ```
3. Keep model names in `config.py` as:
   ```python
   OPENAI_CHAT_MODEL = "qwen/Qwen2.5-7B-Instruct-Q8_0"
   OPENAI_EMBED_MODEL = "text-embedding-qwen3-0.6b-text-embedding"
   ```

### Using OpenAI API

1. Update `.env`:
   ```bash
   OPENAI_API_KEY=sk-your-actual-openai-key
   OPENAI_BASE_URL=https://api.openai.com/v1
   ```
2. Update `config.py`:
   ```python
   OPENAI_CHAT_MODEL = "gpt-4"
   OPENAI_EMBED_MODEL = "text-embedding-3-small"
   EMBEDDING_DIM = 1536  # Change to match OpenAI embedding dimension
   ```


### Access Qdrant Dashboard
Open http://localhost:6333/dashboard in your browser.

### Shell Access
```bash
# Access rag-app container
docker-compose run --rm rag-app /bin/bash

# Access qdrant container
docker exec -it fh-wedel-qdrant /bin/sh
```

### Inspect Document Chunks
```bash
docker-compose run --rm rag-app python inspect_chunks.py
```

## Data Persistence

Data is persisted in the following locations:

- **Docker Volumes:**
  - `qdrant_data` - Vector embeddings and indices
  - `ollama_data` - Downloaded LLM models

- **Bind Mounts:**
  - `./rag-code/data_fh_wedel/` - Downloaded PDFs
  - `./rag-code/fhwedel_docs.json` - Document metadata
  - `./rag-code/metadata_cache.json` - Query expansion cache

### Backup Data
```bash
# Backup Qdrant data
docker run --rm -v fh-wedel_qdrant_data:/data -v $(pwd):/backup ubuntu tar czf /backup/qdrant_backup.tar.gz /data

# Backup Ollama models
docker run --rm -v fh-wedel_ollama_data:/data -v $(pwd):/backup ubuntu tar czf /backup/ollama_backup.tar.gz /data
```

### Clean Up
```bash
# Stop all services
docker-compose down

# Remove volumes (WARNING: deletes all data)
docker-compose down -v

# Remove images
docker-compose down --rmi all
```

## GPU Support

### NVIDIA GPU (Linux)

1. Install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

2. Uncomment GPU sections in `docker-compose.yml`:
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: 1
             capabilities: [gpu]
   ```

3. Restart services:
   ```bash
   docker-compose up -d
   ```

### Apple Silicon / AMD GPUs

Currently, PyTorch GPU support in Docker is limited on macOS. Consider running natively or using CPU inference.

## Development Workflow

For active development:

1. **Edit Code Locally** - Changes in `./rag-code/` are immediately reflected (volume mount)

2. **Hot Reload** - Restart the app without rebuilding:
   ```bash
   docker-compose restart rag-app
   ```

3. **Install New Dependencies:**
   ```bash
   # Add to requirements.txt, then rebuild
   docker-compose build rag-app
   ```

4. **Run Tests:**
   ```bash
   docker-compose run --rm rag-app pytest
   ```

## Alternative: Docker Without Compose

Build and run manually:

```bash
# Build image
docker build -t fh-wedel-rag .

# Run Qdrant
docker run -d --name qdrant -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant

# Run application
docker run -it --rm \
  --link qdrant \
  -e OPENAI_API_KEY=dummy \
  -e OPENAI_BASE_URL=http://host.docker.internal:1234/v1 \
  -e QDRANT_URL=http://qdrant:6333 \
  -v $(pwd)/rag-code/data_fh_wedel:/app/rag-code/data_fh_wedel \
  fh-wedel-rag
```

