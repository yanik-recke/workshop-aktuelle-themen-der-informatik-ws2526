from typing import List, Tuple

from config import MAX_MEMORY_TURNS, TOP_K, OPENAI_CHAT_MODEL, USE_LLM_RERANK
from context_detection import detect_context
from query_expansion import expand_query_with_llm
from memory_utils import create_conversation_buffer
from rag_pipeline import build_filters_from_context, PROMPT_TEMPLATE
from hybrid_retrieval import hybrid_search
from reranker import rerank_with_metadata_boost

from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret


def run_single_turn(query: str) -> Tuple[str, List[str]]:
    """
    Execute one RAG turn and return:
      - answer: str
      - retrieval_context: List[str] (contents of retrieved docs)
    """
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
    expanded_query, original_query, keywords = expand_query_with_llm(
        query, memory_summary, conversation_buffer
    )

    # Build optional filters (currently returns {})
    filters = build_filters_from_context(context_state)
    # Use hybrid retrieval to match current pipeline behavior
    documents = hybrid_search(
        query=expanded_query,
        top_k=TOP_K * 2,  # Fetch more for reranking
        filters=filters,
        original_query=original_query,
        keywords=keywords,
    )
    
    # Rerank chunks using LLM for better relevance
    documents = rerank_with_metadata_boost(
        query=original_query,
        documents=documents,
        top_k=TOP_K,
        use_llm_rerank=USE_LLM_RERANK,
    )

    # Build prompt and generate using the same template as the pipeline
    prompt_builder = PromptBuilder(template=PROMPT_TEMPLATE)
    prompt_result = prompt_builder.run(
        documents=documents,
        query=original_query,
        memory_summary=memory_summary,
    )

    generator = OpenAIGenerator(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        api_base_url=None,  # Force using OpenAI API
        model=OPENAI_CHAT_MODEL,
        timeout=60.0,
        generation_kwargs={"temperature": 0.3, "max_tokens": 512},
    )
    gen_result = generator.run(prompt=prompt_result["prompt"])
    replies = gen_result.get("replies", [])
    answer = replies[0].strip() if replies else "I don't know."

    # Prepare retrieval_context as plain strings (Markdown is fine)
    retrieval_context: List[str] = []
    for doc in documents or []:
        text = getattr(doc, "content", None)
        if isinstance(text, str):
            retrieval_context.append(text)

    return answer, retrieval_context


def retrieve_chunks(query: str) -> List[str]:
    context_state = {
        "degree": [],
        "program": [],
        "doctype": [],
        "status": [],
    }
    memory_summary = ""
    conversation_buffer = create_conversation_buffer()

    context_state = detect_context(query, context_state)
    expanded_query, original_query, keywords = expand_query_with_llm(
        query, memory_summary, conversation_buffer
    )
    filters = build_filters_from_context(context_state)

    documents = hybrid_search(
        query=expanded_query,
        top_k=TOP_K * 2,  # Fetch more for reranking
        filters=filters,
        original_query=original_query,
        keywords=keywords,
    )
    
    # Rerank chunks using LLM for better relevance
    documents = rerank_with_metadata_boost(
        query=original_query,
        documents=documents,
        top_k=TOP_K,
        use_llm_rerank=USE_LLM_RERANK,
    )

    out: List[str] = []
    for doc in documents or []:
        text = getattr(doc, "content", None)
        if isinstance(text, str):
            out.append(text)
    return out
