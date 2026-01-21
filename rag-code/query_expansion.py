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

# German stopwords to filter from keywords
GERMAN_STOPWORDS = {
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer", "eines", "einem", "einen",
    "und", "oder", "aber", "doch", "sondern", "weil", "dass", "wenn", "als", "ob",
    "ich", "du", "er", "sie", "es", "wir", "ihr", "mich", "dich", "sich", "uns", "euch",
    "mir", "dir", "ihm", "ihr", "ihnen", "mein", "dein", "sein", "unser", "euer",
    "ist", "sind", "war", "waren", "bin", "bist", "sein", "haben", "hat", "hatte", "hatten",
    "wird", "werden", "wurde", "wurden", "kann", "kannst", "können", "konnte", "konnten",
    "muss", "musst", "müssen", "musste", "mussten", "soll", "sollst", "sollen", "sollte",
    "will", "willst", "wollen", "wollte", "darf", "darfst", "dürfen", "durfte",
    "was", "wer", "wie", "wo", "wann", "warum", "welche", "welcher", "welches",
    "für", "mit", "bei", "von", "aus", "nach", "zu", "zur", "zum", "über", "unter",
    "vor", "hinter", "neben", "zwischen", "durch", "gegen", "ohne", "um",
    "auf", "an", "in", "im", "am",
    "nicht", "kein", "keine", "keiner", "keines", "keinem", "keinen",
    "auch", "noch", "schon", "nur", "sehr", "mehr", "viel", "wenig",
    "hier", "dort", "da", "dann", "nun", "jetzt", "immer", "nie", "oft",
    "ja", "nein", "denn", "also", "so", "selbst", "ganz", "gar",
    "dieser", "diese", "dieses", "diesem", "diesen", "jener", "jene", "jenes",
    "alle", "alles", "allem", "allen", "aller", "andere", "anderer", "anderen",
    "einige", "einiger", "einigen", "manche", "mancher", "manchen",
    "welch", "solch", "solche", "solcher", "solchen",
    "etwas", "nichts", "jemand", "niemand", "man",
    "gibt", "geben", "gibt's", "habe", "hab", "bitte", "danke",
    "sagen", "sagen?", "sagst", "sagt", "erzählen", "erzähl", "erklären", "erkläre",
    "mir", "mich", "dich", "uns",
}


def filter_stopwords(keywords: List[str]) -> List[str]:
    """Filter out German stopwords, split phrases, and clean keywords."""
    filtered = []
    seen = set()
    
    for kw in keywords:
        # Split multi-word phrases into individual words
        words = kw.split()
        for word in words:
            # Clean punctuation and lowercase
            clean = word.lower().strip('.,;:!?"\'()')
            # Skip stopwords and very short words, avoid duplicates
            if clean not in GERMAN_STOPWORDS and len(clean) > 2 and clean not in seen:
                filtered.append(clean)
                seen.add(clean)
    
    return filtered


def _get_query_llm() -> OpenAIGenerator:
    return OpenAIGenerator(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        model=OPENAI_CHAT_MODEL,
        generation_kwargs={"temperature": 0.2, "max_tokens": 512},  # Increased to avoid truncation
    )


def _history_to_text(history: Deque[Tuple[str, str]], max_turns: int = 4) -> str:
    recent = list(history)[-max_turns:]
    return "\n".join(f"{role}: {msg}" for role, msg in recent)


def _truncate_chars(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]..."


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
    # Additional hard caps to avoid model context overflow
    recent_history = _truncate_chars(recent_history, 2000)
    mem_short = _truncate_chars(memory_summary, 2000)
    hints_text = _truncate_chars(hints_text, 3000)

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
{mem_short or "None"}

User query:
{query}

Return ONLY valid JSON:
{{
  "original_query": "<original question>",
  "expanded_query": "<expanded retrieval query>",
  "keywords": ["<keyword1>", "<keyword2>", ...]
}}

The "keywords" array should contain:
- Key terms from the original query (corrected for spelling)
- German synonyms for important concepts
- Related technical terms
- At least 3-8 keywords for good retrieval
"""

    try:
        resp = llm.run(prompt)
        raw = resp["replies"][0].strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # try to extract JSON object substring
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = raw[start:end+1]
                data = json.loads(candidate)
            else:
                raise

        expanded_query = (data.get("expanded_query") or query).strip()
        original_query = (data.get("original_query") or query).strip()
        keywords = data.get("keywords", [])
        if not keywords:
            # Fallback: extract keywords from expanded query
            keywords = [w.lower() for w in expanded_query.split() if len(w) > 3]
        # Filter stopwords from keywords
        keywords = filter_stopwords(keywords)
        return expanded_query, original_query, keywords

    except Exception as e:
        print(f"Query expansion failed: {e}")
        # Fallback keywords from original query (filtered)
        fallback_keywords = filter_stopwords([w for w in query.split()])
        return f"{query} an der FH Wedel", query, fallback_keywords
