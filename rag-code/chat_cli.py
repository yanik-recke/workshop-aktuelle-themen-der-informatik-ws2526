import textwrap
from typing import Dict, List, Tuple

from config import MAX_MEMORY_TURNS, OPENAI_EMBED_MODEL, TOP_K
from context_detection import detect_context
from query_expansion import expand_query_with_llm
from memory_utils import update_memory_summary, create_conversation_buffer
from rag_pipeline import create_rag_pipeline, build_filters_from_context, run_rag_query, run_comparison_query
from hybrid_retrieval import hybrid_search, print_retrieval_debug
from comparison_handler import handle_comparison_query
from document_store import get_document_store


def format_context(context_state: Dict[str, List[str]]) -> str:
    parts = []
    for key in ["degree", "program", "doctype", "status"]:
        vals = context_state.get(key)
        if not vals:
            continue
        if isinstance(vals, str):
            vals = [vals]
        parts.append(f"{key}=" + ", ".join(vals))
    return " | ".join(parts)


def chat_loop():
    print("🧠 FH-Wedel RAG-Chatbot (Haystack) bereit! Tippe 'exit' zum Beenden.\n")

    rag_pipeline = create_rag_pipeline()

    context_state: Dict[str, List[str]] = {
        "degree": [],
        "program": [],
        "doctype": [],
        "status": [],
    }
    memory_summary = ""
    conversation_buffer = create_conversation_buffer()
    debug = False

    while True:
        try:
            query = input("👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Tschüss!")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("👋 Tschüss!")
            break

        if query.lower() == "show context":
            print(f"🧩 Active context: {format_context(context_state) or 'none'}\n")
            continue

        if query.lower() == "clear context":
            context_state = {
                "degree": [],
                "program": [],
                "doctype": [],
                "status": [],
            }
            print("🧹 Kontext zurückgesetzt.\n")
            continue

        if query.lower() == "debug on":
            debug = True
            print("Debug-Modus: AN\n")
            continue

        if query.lower() == "debug off":
            debug = False
            print("Debug-Modus: AUS\n")
            continue

        context_state = detect_context(query, context_state)

        expanded_query, original_query, keywords = expand_query_with_llm(
            query,
            memory_summary,
            conversation_buffer,
        )
        
        print(f"Keywords: {keywords}")

        print(f"🧩 Active context: {format_context(context_state) or 'none'}")

        filters = build_filters_from_context(context_state)

        conversation_history = "\n".join(
            f"{role}: {msg}" for role, msg in conversation_buffer
        )
        # hard caps to safeguard model context window
        mem_short = memory_summary[:3000]
        hist_short = conversation_history[:4000]

        # Check if this is a comparison query
        comparison_result = handle_comparison_query(
            query=original_query,
            expanded_query=expanded_query,
            top_k=TOP_K,
        )
        
        if comparison_result:
            # Handle comparison query
            docs, entity1, entity2 = comparison_result
            if debug:
                print(f"\nComparison: '{entity1}' vs '{entity2}'")
                print_retrieval_debug(docs, original_query)
            try:
                answer = run_comparison_query(
                    original_query=original_query,
                    documents=docs,
                    entity1=entity1,
                    entity2=entity2,
                    memory_summary=mem_short,
                )
            except Exception as e:
                print(f"Comparison failed: {e}")
                continue
        elif not debug:
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
                print(f"Generation failed: {e}")
                continue
        else:
            # Use hybrid search (keyword + semantic) for better retrieval
            # Pass enhanced keywords for keyword matching, expanded_query for semantics
            try:
                retrieved = hybrid_search(
                    query=expanded_query,
                    top_k=TOP_K,
                    filters=filters,
                    original_query=original_query,
                    keywords=keywords,  # Use enhanced keywords for search
                )
            except Exception as e:
                print(f"Retrieval failed: {e}")
                continue

            # Print debug info with keyword match indicators
            print_retrieval_debug(retrieved, original_query)

            # Now also generate an answer using the full pipeline
            print("\nGenerating answer...")
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
                answer = f"[Debug] Generation failed: {e}"

        conversation_buffer.append(("user", query))
        conversation_buffer.append(("assistant", answer))

        if len(conversation_buffer) >= MAX_MEMORY_TURNS:
            memory_summary = update_memory_summary(memory_summary, conversation_buffer)
            conversation_buffer.clear()

        print("\n🤖 Assistant:\n" + textwrap.fill(answer, width=100))
        print("\n" + "=" * 100 + "\n")


if __name__ == "__main__":
    chat_loop()
