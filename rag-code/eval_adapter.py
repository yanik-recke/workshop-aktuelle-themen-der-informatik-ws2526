from typing import List, Tuple

from config import MAX_MEMORY_TURNS
from context_detection import detect_context
from query_expansion import expand_query_with_llm
from memory_utils import create_conversation_buffer
from rag_pipeline import create_rag_pipeline, build_filters_from_context


def run_single_turn(query: str) -> Tuple[str, List[str]]:
    """
    Execute one RAG turn and return:
      - answer: str
      - retrieval_context: List[str] (contents of retrieved docs)
    """
    # Create components
    rag_pipeline = create_rag_pipeline()

    # Context and memory state
    context_state = {
        "degree": [],
        "program": [],
        "doctype": [],
        "status": [],
    }
    memory_summary = ""
    conversation_buffer = create_conversation_buffer()

    # Detect context and expand query
    context_state = detect_context(query, context_state)
    expanded_query, original_query = expand_query_with_llm(
        query, memory_summary, conversation_buffer
    )

    # Build optional filters (currently returns {})
    filters = build_filters_from_context(context_state)

    # Assemble pipeline inputs mirroring run_rag_query, but we capture retriever docs
    inputs = {
        "text_embedder": {"text": expanded_query},
        "retriever": {},
        "prompt_builder": {
            "query": original_query,
            "memory_summary": memory_summary,
            "conversation_history": "",
        },
        "generator": {},
    }
    if filters:
        inputs["retriever"]["filters"] = filters

    result = rag_pipeline.run(inputs)

    # Extract generator reply
    replies = result.get("generator", {}).get("replies", [])
    answer = replies[0].strip() if replies else "I don't know."

    # Extract retriever documents for retrieval_context
    documents = result.get("retriever", {}).get("documents", []) or []
    retrieval_context: List[str] = []
    try:
        for doc in documents:
            # Haystack Document exposes .content
            text = getattr(doc, "content", None)
            if isinstance(text, str):
                retrieval_context.append(text)
    except Exception:
        pass

    return answer, retrieval_context
