# FH Wedel RAG Chatbot - REST API Documentation

## Quick Start

### Start the API Server
```bash
# Using docker-compose (recommended)
docker-compose up -d rag-api

# Or using Make
make api
```

The API will be available at: **http://localhost:8000**

### Interactive API Documentation

FastAPI provides automatic interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### 1. Root Endpoint

**GET /**

Returns API information and available endpoints.

```bash
curl http://localhost:8000/
```

Response:
```json
{
  "name": "FH Wedel RAG Chatbot API",
  "version": "1.0.0",
  "endpoints": {
    "POST /chat": "Send a query to the chatbot",
    "GET /context/{session_id}": "Get current context for a session",
    ...
  }
}
```

---

### 2. Health Check

**GET /health**

Check if the API and RAG pipeline are healthy.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-01T12:00:00",
  "rag_pipeline_initialized": true
}
```

---

### 3. Chat Query

**POST /chat**

Send a query to the RAG chatbot.

**Request Body:**
```json
{
  "query": "Welche Module sind im ersten Semester Informatik?",
  "session_id": "optional-session-id",
  "debug": false,
  "context": null
}
```

**Parameters:**
- `query` (required): The user's question
- `session_id` (optional): Session ID for conversation continuity. If not provided, a new session is created.
- `debug` (optional): Enable debug mode to see retrieval details (default: false)
- `context` (optional): Manual context override (degree, program, doctype, status)

**Example with cURL:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Welche Module sind im ersten Semester Informatik?"
  }'
```

**Response:**
```json
{
  "answer": "Im ersten Semester des Informatik-Studiums...",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "context": {
    "degree": ["Bachelor"],
    "program": ["Informatik"],
    "doctype": [],
    "status": []
  },
  "keywords": ["Module", "ersten", "Semester", "Informatik"],
  "debug_info": null
}
```

**Response Fields:**
- `answer`: The generated answer to the query
- `session_id`: Session ID for this conversation (use in subsequent requests)
- `context`: Active context detected/used for retrieval
- `keywords`: Extracted keywords from the query
- `debug_info`: Debug information (only if `debug=true`)

---

### 4. Continue Conversation

To maintain conversation context, use the `session_id` from the previous response:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Und im zweiten Semester?",
    "session_id": "550e8400-e29b-41d4-a716-446655440000"
  }'
```

The chatbot will remember the previous context (Informatik, Bachelor) and answer accordingly.

---

### 5. Debug Mode

Enable debug mode to see retrieved documents and scores:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Was ist die Regelstudienzeit?",
    "debug": true
  }'
```

**Response with debug_info:**
```json
{
  "answer": "Die Regelstudienzeit beträgt...",
  "session_id": "...",
  "context": {...},
  "keywords": [...],
  "debug_info": {
    "type": "hybrid",
    "expanded_query": "Was ist die Regelstudienzeit im Studium?",
    "keywords": ["Regelstudienzeit"],
    "num_documents": 5,
    "documents": [
      {
        "content": "Die Regelstudienzeit für den Bachelor...",
        "score": 0.92,
        "metadata": {
          "degree": "Bachelor",
          "program": "Informatik",
          "doctype": "Studien- und Prüfungsordnung"
        }
      }
    ]
  }
}
```

---

### 6. Get Session Context

**GET /context/{session_id}**

Retrieve the current context for a session.

```bash
curl http://localhost:8000/context/550e8400-e29b-41d4-a716-446655440000
```

Response:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "context": {
    "degree": ["Bachelor"],
    "program": ["Informatik"],
    "doctype": [],
    "status": []
  }
}
```

---

### 7. Clear Session Context

**DELETE /context/{session_id}**

Reset the context for a session (useful when switching topics).

```bash
curl -X DELETE http://localhost:8000/context/550e8400-e29b-41d4-a716-446655440000
```

Response:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "context": {
    "degree": [],
    "program": [],
    "doctype": [],
    "status": []
  }
}
```

---

### 8. Delete Session

**DELETE /session/{session_id}**

Delete a session and all its data.

```bash
curl -X DELETE http://localhost:8000/session/550e8400-e29b-41d4-a716-446655440000
```

Response:
```json
{
  "message": "Session 550e8400-e29b-41d4-a716-446655440000 deleted successfully"
}
```

---

### 9. List All Sessions

**GET /sessions**

List all active sessions (useful for monitoring/debugging).

```bash
curl http://localhost:8000/sessions
```

Response:
```json
[
  {
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "created_at": "2025-01-01T10:00:00",
    "message_count": 5,
    "context": {
      "degree": ["Bachelor"],
      "program": ["Informatik"],
      "doctype": [],
      "status": []
    }
  }
]
```

---

## Advanced Usage

### Manual Context Override

You can manually set the context instead of relying on automatic detection:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Welche Module gibt es?",
    "context": {
      "degree": ["Master"],
      "program": ["Data Science & Artificial Intelligence"],
      "doctype": ["Modulhandbuch"],
      "status": ["aktuell"]
    }
  }'
```

---

### Comparison Queries

The API automatically detects comparison queries:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Was ist der Unterschied zwischen Bachelor Informatik und Wirtschaftsinformatik?"
  }'
```

Response will include a structured comparison.

---

## Client Examples

### Python Client

```python
import requests

API_URL = "http://localhost:8000"

# Start a conversation
response = requests.post(
    f"{API_URL}/chat",
    json={"query": "Was ist die Regelstudienzeit für Informatik?"}
)
data = response.json()

print(f"Answer: {data['answer']}")
session_id = data['session_id']

# Continue conversation
response = requests.post(
    f"{API_URL}/chat",
    json={
        "query": "Und wie viele ECTS sind das?",
        "session_id": session_id
    }
)
print(f"Answer: {response.json()['answer']}")
```

---

### JavaScript Client

```javascript
const API_URL = 'http://localhost:8000';

async function chat(query, sessionId = null) {
  const response = await fetch(`${API_URL}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query: query,
      session_id: sessionId,
    }),
  });
  return await response.json();
}

// Example usage
const result = await chat('Welche Module sind im ersten Semester?');
console.log(result.answer);

// Continue conversation
const nextResult = await chat('Und im zweiten?', result.session_id);
console.log(nextResult.answer);
```

---

### cURL Multi-Query Example

```bash
#!/bin/bash

# First query
RESPONSE=$(curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Welche Studiengänge gibt es im Bachelor?"}')

echo "Answer: $(echo $RESPONSE | jq -r '.answer')"
SESSION_ID=$(echo $RESPONSE | jq -r '.session_id')

# Follow-up query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"Erzähl mir mehr über Informatik\", \"session_id\": \"$SESSION_ID\"}" \
  | jq -r '.answer'
```

---

## Session Management

### Session Lifecycle

1. **Creation**: Sessions are created automatically when a query is sent without a `session_id`.
2. **Persistence**: Sessions persist in memory for 1 hour (configurable via `SESSION_TIMEOUT` in `api.py`).
3. **Cleanup**: Old sessions are automatically cleaned up during health checks.
4. **Deletion**: Sessions can be manually deleted via `DELETE /session/{session_id}`.

### Session Data

Each session stores:
- `context_state`: Detected context (degree, program, doctype, status)
- `memory_summary`: Compressed conversation history
- `conversation_buffer`: Recent conversation turns (up to 5)
- `created_at`: Session creation timestamp
- `last_accessed`: Last activity timestamp
- `message_count`: Total number of messages exchanged

---

## Production Considerations

### Scalability

For production deployments, consider:

1. **Persistent Session Storage**: Replace in-memory sessions with Redis
   ```python
   # In api.py, replace sessions dict with Redis client
   import redis
   session_store = redis.Redis(host='redis', port=6379)
   ```

2. **Load Balancing**: Run multiple API instances behind a load balancer
   ```yaml
   # docker-compose.yml
   rag-api:
     deploy:
       replicas: 3
   ```

3. **Rate Limiting**: Add rate limiting middleware
   ```python
   from slowapi import Limiter
   limiter = Limiter(key_func=get_remote_address)
   ```

4. **Authentication**: Add API key authentication
   ```python
   from fastapi.security import APIKeyHeader
   api_key_header = APIKeyHeader(name="X-API-Key")
   ```

### CORS Configuration

For production, restrict allowed origins:

```python
# In api.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Monitoring

### Health Checks

Kubernetes/Docker Swarm health check:
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```

### Logging

API logs are written to stdout. View with:
```bash
docker-compose logs -f rag-api
```

---


## Example Web Interface

See `client.html` for a simple web interface to interact with the API. For Test until we have a real frontend.

---

## Development

### Run API Locally (without Docker)

```bash
cd rag-code
pip install -r requirements.txt -r requirements-api.txt
export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://localhost:1234/v1
export QDRANT_URL=http://localhost:6333
uvicorn api:app --reload
```
---