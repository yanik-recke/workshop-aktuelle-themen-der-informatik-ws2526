# Frontend Application

## Prerequisites

**Option 1: Docker (Recommended)**
- Docker and Docker Compose

**Option 2: Local Development**
- Node.js 20.9.0+ and npm
- MongoDB (local installation or Docker container)

## Setup

_Disclaimer_: The frontend takes a while to compile inside the container...

### Option 1: Docker Setup (Recommended)

Run everything with Docker Compose from the project root:

```bash
# Start all services (frontend, backend, MongoDB, Qdrant)
docker compose up -d

# Or run with logs
docker compose up
```

The application will be available at `http://localhost:3000`

The Docker setup automatically configures:
- MongoDB connection to the containerized MongoDB
- RAG API connection to the containerized backend
- Hot reload for development (code changes reflect immediately)

### Option 2: Local Development Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Configure environment variables:
   - Copy `.env.local.example` to `.env.local`
   - Update the following variables:

   For local development with Docker services:
   ```env
   MONGODB_URI=mongodb://admin:password@localhost:27017
   NEXT_PUBLIC_RAG_API_URL=http://localhost:8000
   ```

3. Make sure the RAG API and MongoDB services are running:
   ```bash
   # From the project root directory
   docker compose up rag-api mongodb -d
   ```

4. Start the development server:
   ```bash
   npm run dev
   ```

The application will be available at `http://localhost:3000`

## Production Build

```bash
npm run build
npm start
```

## MongoDB Collections

The application uses the following MongoDB collections in the `chatbot` database:

- `chats`: Stores chat conversations with messages

## Features

- Chat history persistence in MongoDB
- Integration with RAG API backend (rag-code/api.py)
- Session-based conversation continuity with RAG API
- Real-time chat updates
- Chat deletion with MongoDB sync
- Model selection interface (Gemini, ChatGPT, Llama)