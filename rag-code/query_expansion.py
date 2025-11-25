# query_expansion.py
import json
from collections import deque
from typing import Deque, Tuple, Dict, List

from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret

from config import OPENAI_CHAT_MODEL

# -------------------------------------------------------
# Load Metadata Cache (all extracted metadata)
# -------------------------------------------------------

def load_metadata_cache() -> Dict:
    try:
        with open("metadata_cache.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

METADATA = load_metadata_cache()


# -------------------------------------------------------
# Helper functions
# -------------------------------------------------------

def _get_query_llm() -> OpenAIGenerator:
    return OpenAIGenerator(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        model=OPENAI_CHAT_MODEL,
        generation_kwargs={"temperature": 0.2},
    )


def _history_to_text(history: Deque[Tuple[str, str]], max_turns: int = 4) -> str:
    recent = list(history)[-max_turns:]
    return "\n".join(f"{role}: {msg}" for role, msg in recent)


def build_metadata_hints() -> List[str]:
    hints = []

    for key, values in METADATA.items():
        if isinstance(values, list):
            for v in values:
                if isinstance(v, str) and len(v) < 80:
                    hints.append(v)

    return sorted(set(hints))


METADATA_HINTS = build_metadata_hints()


# -------------------------------------------------------
# Query Expansion with Context Understanding
# -------------------------------------------------------

def expand_query_with_llm(
    query: str,
    memory_summary: str,
    conversation_buffer: Deque[Tuple[str, str]],
) -> Tuple[str, str]:

    llm = _get_query_llm()
    recent_history = _history_to_text(conversation_buffer)

    hints_text = ", ".join(METADATA_HINTS[:200])  # cap for safety

    prompt = f"""
You rewrite user queries to improve semantic retrieval for a RAG system of the FH Wedel.

Your tasks:
1. Detect relevant context from:
   - the original question
   - the conversation history
   - the memory summary
   - the allowed metadata hints (list below)

2. Expand the query ONLY by adding:
   - synonyms
   - related keywords
   - module names
   - module IDs
   - programs
   - degrees
   - semesters
   - topics
   BUT ONLY if they appear in:
      • the metadata-hints list
      • the user's question
      • the conversation history
      • the memory summary

3. You MUST NOT invent any new facts, module names, numbers, rules or details.

4. Always scope the expanded query to “FH Wedel”.

5. Keep the language of the user (usually German).

6. Output must be retrieval-focused and as exact as possible. 
   No answering. No explanations.

Allowed metadata keywords:
{hints_text}

Conversation history:
{recent_history or "None"}

Memory summary:
{memory_summary or "None"}

User query:
{query}

Return ONLY valid JSON:
{{
  "original_query": "<original question>",
  "expanded_query": "<expanded retrieval query>"
}}
"""

    try:
        resp = llm.run(prompt)
        raw = resp["replies"][0].strip()
        data = json.loads(raw)

        expanded_query = (data.get("expanded_query") or query).strip()
        original_query = (data.get("original_query") or query).strip()



        return expanded_query, original_query

    except Exception as e:
        print(f"⚠️ Query expansion failed: {e}")
        return f"{query} an der FH Wedel", query
