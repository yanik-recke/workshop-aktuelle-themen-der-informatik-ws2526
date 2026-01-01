# Frontend Application

## Prerequisites

- Node.js 18+ and npm
- MongoDB (local installation or Docker container)

## Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Configure environment variables:
   - Copy `.env.local.example` to `.env.local`
   - Update the following variables:
     - `MONGODB_URI`: MongoDB connection string
     - `NEXT_PUBLIC_RAG_API_URL`: RAG API endpoint URL

   For Docker (using the docker-compose.yml in the root):
   ```env
   MONGODB_URI=mongodb://admin:password@localhost:27017
   NEXT_PUBLIC_RAG_API_URL=http://localhost:8000
   ```

## Running the Application

**Prerequisites:** Make sure the RAG API and MongoDB services are running:
```bash
# From the project root directory
docker compose up rag-api mongodb -d
```

Development mode:
```bash
npm run dev
```

Production build:
```bash
npm run build
npm start
```

The application will be available at `http://localhost:3000`

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