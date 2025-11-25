import textwrap
from typing import Dict, List, Tuple

from config import MAX_MEMORY_TURNS
from context_detection import detect_context
from query_expansion import expand_query_with_llm
from memory_utils import update_memory_summary, create_conversation_buffer
from rag_pipeline import create_rag_pipeline, build_filters_from_context, run_rag_query


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

        context_state = detect_context(query, context_state)

        expanded_query, original_query = expand_query_with_llm(
            query,
            memory_summary,
            conversation_buffer,
        )

        print(f"🧩 Active context: {format_context(context_state) or 'none'}")

        filters = build_filters_from_context(context_state)

        conversation_history = "\n".join(
            f"{role}: {msg}" for role, msg in conversation_buffer
        )

        answer = run_rag_query(
            rag_pipeline,
            original_query=original_query,
            expanded_query=expanded_query,
            memory_summary=memory_summary,
            conversation_history=conversation_history,
            filters=filters,
        )

        conversation_buffer.append(("user", query))
        conversation_buffer.append(("assistant", answer))

        if len(conversation_buffer) >= MAX_MEMORY_TURNS:
            memory_summary = update_memory_summary(memory_summary, conversation_buffer)
            conversation_buffer.clear()

        print("\n🤖 Assistant:\n" + textwrap.fill(answer, width=100))
        print("\n" + "=" * 100 + "\n")


if __name__ == "__main__":
    chat_loop()
