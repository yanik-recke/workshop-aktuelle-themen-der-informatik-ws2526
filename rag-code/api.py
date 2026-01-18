"""
FastAPI REST API for FH Wedel RAG Chatbot
Provides HTTP endpoints for querying the RAG system.
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from collections import deque
import uuid
from datetime import datetime
import logging

from config import MAX_MEMORY_TURNS, TOP_K
from context_detection import detect_context
from query_expansion import expand_query_with_llm
from memory_utils import update_memory_summary, create_conversation_buffer
from rag_pipeline import create_rag_pipeline, build_filters_from_context, run_rag_query, run_comparison_query
from hybrid_retrieval import hybrid_search, print_retrieval_debug
from comparison_handler import handle_comparison_query

# NeMo Guardrails
from nemoguardrails import RailsConfig, LLMRails

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FH Wedel RAG Chatbot API",
    description="REST API for querying FH Wedel study documents using RAG",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG pipeline (initialized once)
rag_pipeline = None

# Global Guardrails instance
guardrails: Optional[LLMRails] = None

# Session storage (in-memory, use Redis in production)
sessions: Dict[str, dict] = {}

# Session timeout (in seconds)
SESSION_TIMEOUT = 3600  # 1 hour


# ============================================================================
# Pydantic Models
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for chat queries"""
    query: str = Field(..., description="User query text", min_length=1)
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    debug: bool = Field(False, description="Enable debug mode for retrieval details")
    context: Optional[Dict[str, List[str]]] = Field(
        None,
        description="Manual context override (degree, program, doctype, status)"
    )


class QueryResponse(BaseModel):
    """Response model for chat queries"""
    answer: str = Field(..., description="Generated answer")
    session_id: str = Field(..., description="Session ID for this conversation")
    context: Dict[str, List[str]] = Field(..., description="Active context used for retrieval")
    keywords: List[str] = Field(..., description="Extracted keywords from query")
    debug_info: Optional[Dict] = Field(None, description="Debug information (if debug=True)")


class ContextResponse(BaseModel):
    """Response model for context queries"""
    session_id: str
    context: Dict[str, List[str]]


class SessionResponse(BaseModel):
    """Response model for session info"""
    session_id: str
    created_at: str
    message_count: int
    context: Dict[str, List[str]]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    rag_pipeline_initialized: bool


# ============================================================================
# Session Management
# ============================================================================

def get_or_create_session(session_id: Optional[str] = None) -> tuple[str, dict]:
    """Get existing session or create a new one"""
    if session_id and session_id in sessions:
        session = sessions[session_id]
        session["last_accessed"] = datetime.now()
        return session_id, session

    # Create new session
    new_session_id = str(uuid.uuid4())
    sessions[new_session_id] = {
        "context_state": {
            "degree": [],
            "program": [],
            "doctype": [],
            "status": [],
        },
        "memory_summary": "",
        "conversation_buffer": create_conversation_buffer(),
        "created_at": datetime.now(),
        "last_accessed": datetime.now(),
        "message_count": 0,
    }
    logger.info(f"Created new session: {new_session_id}")
    return new_session_id, sessions[new_session_id]


def cleanup_old_sessions():
    """Remove sessions older than SESSION_TIMEOUT"""
    now = datetime.now()
    to_remove = []
    for sid, session in sessions.items():
        if (now - session["last_accessed"]).seconds > SESSION_TIMEOUT:
            to_remove.append(sid)

    for sid in to_remove:
        del sessions[sid]
        logger.info(f"Cleaned up expired session: {sid}")


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline and Guardrails on startup"""
    global rag_pipeline, guardrails
    logger.info("Initializing RAG pipeline...")
    try:
        rag_pipeline = create_rag_pipeline()
        logger.info("RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        raise

    # Initialize NeMo Guardrails (optional, controlled by ENABLE_GUARDRAILS env var)
    import os
    if os.getenv("ENABLE_GUARDRAILS", "false").lower() == "true":
        logger.info("Initializing NeMo Guardrails...")
        try:
            config = RailsConfig.from_path("guardrails")
            guardrails = LLMRails(config)
            logger.info("NeMo Guardrails initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Guardrails (continuing without): {e}")
            guardrails = None
    else:
        logger.info("NeMo Guardrails disabled (set ENABLE_GUARDRAILS=true to enable)")
        guardrails = None


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API...")
    sessions.clear()


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "name": "FH Wedel RAG Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "POST /chat": "Send a query to the chatbot",
            "GET /context/{session_id}": "Get current context for a session",
            "DELETE /context/{session_id}": "Clear context for a session",
            "DELETE /session/{session_id}": "Delete a session",
            "GET /sessions": "List all active sessions",
            "GET /health": "Health check",
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    cleanup_old_sessions()
    return HealthResponse(
        status="healthy" if rag_pipeline else "unhealthy",
        timestamp=datetime.now().isoformat(),
        rag_pipeline_initialized=rag_pipeline is not None,
    )


@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    """
    Main chat endpoint for querying the RAG system.

    - **query**: The user's question
    - **session_id**: Optional session ID for conversation continuity
    - **debug**: Enable debug mode to see retrieval details
    - **context**: Manual context override (optional)
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    # Check input with guardrails (input validation only)
    if guardrails:
        try:
            # Use guardrails to check if input is appropriate
            # We check the response for blocking indicators
            input_check = await guardrails.generate_async(
                messages=[{"role": "user", "content": request.query}]
            )
            if input_check and input_check.get("content"):
                guardrail_response = input_check["content"]
                # Only block if guardrails explicitly refuses (off-topic detection)
                block_indicators = [
                    "ich bin ein assistent speziell für fragen zum studium",
                    "off-topic",
                    "kann ich nicht beantworten",
                    "bitte stelle mir fragen zu studiengängen",
                ]
                if any(indicator in guardrail_response.lower() for indicator in block_indicators):
                    logger.info(f"Guardrails blocked input: {request.query}")
                    return QueryResponse(
                        answer="Ich bin ein Assistent speziell für Fragen zum Studium an der FH Wedel. Bitte stelle mir Fragen zu Studiengängen, Modulen oder Prüfungsordnungen.",
                        session_id=request.session_id or str(uuid.uuid4()),
                        context={},
                        keywords=[],
                        debug_info={"guardrails": "input_blocked"} if request.debug else None,
                    )
        except Exception as e:
            logger.warning(f"Guardrails input check failed (continuing without): {e}")

    # Get or create session
    session_id, session = get_or_create_session(request.session_id)

    # Use manual context override if provided
    if request.context:
        context_state = request.context
    else:
        context_state = session["context_state"]

    # Detect context from query
    context_state = detect_context(request.query, context_state)
    session["context_state"] = context_state

    # Expand query with LLM
    try:
        expanded_query, original_query, keywords = expand_query_with_llm(
            request.query,
            session["memory_summary"],
            session["conversation_buffer"],
        )
    except Exception as e:
        logger.error(f"Query expansion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query expansion failed: {str(e)}")

    # Build filters from context
    filters = build_filters_from_context(context_state)

    # Prepare conversation history
    conversation_history = "\n".join(
        f"{role}: {msg}" for role, msg in session["conversation_buffer"]
    )
    mem_short = session["memory_summary"][:3000]
    hist_short = conversation_history[:4000]

    debug_info = None

    # Check if this is a comparison query
    comparison_result = handle_comparison_query(
        query=original_query,
        expanded_query=expanded_query,
        top_k=TOP_K,
    )

    if comparison_result:
        # Handle comparison query
        docs, entity1, entity2 = comparison_result

        if request.debug:
            debug_info = {
                "type": "comparison",
                "entity1": entity1,
                "entity2": entity2,
                "num_documents": len(docs),
                "documents": [
                    {
                        "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                        "score": doc.score,
                        "metadata": doc.meta,
                    }
                    for doc in docs[:5]  # Top 5 for debug
                ]
            }

        try:
            answer = run_comparison_query(
                original_query=original_query,
                documents=docs,
                entity1=entity1,
                entity2=entity2,
                memory_summary=mem_short,
            )
        except Exception as e:
            logger.error(f"Comparison query failed: {e}")
            raise HTTPException(status_code=500, detail=f"Comparison query failed: {str(e)}")

    elif request.debug:
        # Debug mode: retrieve documents and show details
        try:
            retrieved = hybrid_search(
                query=expanded_query,
                top_k=TOP_K,
                filters=filters,
                original_query=original_query,
                keywords=keywords,
            )

            debug_info = {
                "type": "hybrid",
                "expanded_query": expanded_query,
                "keywords": keywords,
                "num_documents": len(retrieved),
                "documents": [
                    {
                        "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                        "score": doc.score,
                        "metadata": doc.meta,
                    }
                    for doc in retrieved[:5]  # Top 5 for debug
                ]
            }
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

        # Generate answer
        try:
            answer = run_rag_query(
                rag_pipeline,
                original_query=original_query,
                expanded_query=expanded_query,
                memory_summary=mem_short,
                conversation_history=hist_short,
                filters=filters,
                keywords=keywords,
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            answer = f"[Debug] Generation failed: {str(e)}"

    else:
        # Normal RAG query - also log retrieved documents
        try:
            # First retrieve documents for logging
            retrieved = hybrid_search(
                query=expanded_query,
                top_k=TOP_K,
                filters=filters,
                original_query=original_query,
                keywords=keywords,
            )

            # Log retrieved documents
            logger.info(f"=== Retrieved {len(retrieved)} documents for query: '{original_query}' ===")
            for i, doc in enumerate(retrieved):
                logger.info(f"  [{i+1}] Score: {doc.score:.4f} | "
                           f"{doc.meta.get('doctype', 'N/A')} | "
                           f"{doc.meta.get('program', 'N/A')} | "
                           f"{doc.meta.get('degree', 'N/A')}")
                logger.info(f"      Content preview: {doc.content[:150]}...")

            answer = run_rag_query(
                rag_pipeline,
                original_query=original_query,
                expanded_query=expanded_query,
                memory_summary=mem_short,
                conversation_history=hist_short,
                filters=filters,
                keywords=keywords,
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    # Update conversation memory
    session["conversation_buffer"].append(("user", request.query))
    session["conversation_buffer"].append(("assistant", answer))
    session["message_count"] += 1

    if len(session["conversation_buffer"]) >= MAX_MEMORY_TURNS:
        session["memory_summary"] = update_memory_summary(
            session["memory_summary"],
            session["conversation_buffer"]
        )
        session["conversation_buffer"].clear()

    return QueryResponse(
        answer=answer,
        session_id=session_id,
        context=context_state,
        keywords=keywords,
        debug_info=debug_info,
    )


@app.get("/context/{session_id}", response_model=ContextResponse)
async def get_context(session_id: str):
    """Get the current context for a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    return ContextResponse(
        session_id=session_id,
        context=sessions[session_id]["context_state"],
    )


@app.delete("/context/{session_id}", response_model=ContextResponse)
async def clear_context(session_id: str):
    """Clear the context for a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    sessions[session_id]["context_state"] = {
        "degree": [],
        "program": [],
        "doctype": [],
        "status": [],
    }

    return ContextResponse(
        session_id=session_id,
        context=sessions[session_id]["context_state"],
    )


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and all its data"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    del sessions[session_id]
    logger.info(f"Deleted session: {session_id}")
    return {"message": f"Session {session_id} deleted successfully"}


@app.get("/sessions", response_model=List[SessionResponse])
async def list_sessions():
    """List all active sessions"""
    cleanup_old_sessions()

    return [
        SessionResponse(
            session_id=sid,
            created_at=session["created_at"].isoformat(),
            message_count=session["message_count"],
            context=session["context_state"],
        )
        for sid, session in sessions.items()
    ]


# ============================================================================
# Run with: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
