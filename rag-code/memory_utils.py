from collections import deque
from typing import Deque, List, Tuple

from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret

from config import OPENAI_CHAT_MODEL, OPENAI_CHAT_BASE_URL, MAX_MEMORY_TURNS


def _get_memory_llm() -> OpenAIGenerator:
    return OpenAIGenerator(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        api_base_url=OPENAI_CHAT_BASE_URL,  # Use OpenAI API for chat
        model=OPENAI_CHAT_MODEL,
        generation_kwargs={"temperature": 0},
    )


def update_memory_summary(
    memory_summary: str,
    conversation_buffer: Deque[Tuple[str, str]],
) -> str:
    """
    Verdichtet den Verlauf in eine kurze, stabile Zusammenfassung
    (Fakten, Präferenzen, Ziele).
    """
    if not conversation_buffer:
        return memory_summary

    convo_text = "\n".join(f"{role}: {msg}" for role, msg in conversation_buffer)
    llm = _get_memory_llm()

    prompt = f"""
Condense the conversation below into a short memory summary.
Keep only stable facts, preferences, constraints and goals.
Ignore greetings, filler, and one-off details.

Existing summary:
{memory_summary or "None"}

Conversation to add:
{convo_text}

Updated summary:
"""
    try:
        resp = llm.run(prompt)
        return resp["replies"][0].strip()
    except Exception as e:
        print(f"⚠️ Memory summarization failed: {e}")
        return memory_summary


def create_conversation_buffer() -> Deque[Tuple[str, str]]:
    return deque(maxlen=MAX_MEMORY_TURNS)
