# Frontend Application

## Prerequisites

- Node.js 18+ and npm
- MongoDB (local installation or Docker container)

## Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Configure MongoDB:
   - Copy `.env.local.example` to `.env.local`
   - Update `MONGODB_URI` with your MongoDB connection string

   For Docker (using the docker-compose.yml in the root):
   ```
   MONGODB_URI=mongodb://admin:password@localhost:27017
   ```

## Running the Application

Development mode:
```bash
npm run dev
```

Production build:
```bash
npm run build
npm start
```

## MongoDB Collections

The application uses the following MongoDB collections in the `chatbot` database:

- `chats`: Stores chat conversations with messages

## Features

- Chat history persistence in MongoDB
- Real-time chat updates
- Chat deletion with MongoDB sync
- Hardcoded AI model selection (Gemini, ChatGPT, Llama)