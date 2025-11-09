import json
import textwrap
from openai import OpenAI
from retriever import retrieve_relevant_chunks
from config import *
from collections import deque

client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
MAX_MEMORY_TURNS = 5   
memory_summary = ""    
conversation_buffer = deque(maxlen=MAX_MEMORY_TURNS)

# -----------------------------------------
# CATEGORY DEFINITIONS
# -----------------------------------------
DEGREES = ["Bachelor", "Master"]
PROGRAMS = [
    "Informatik", "Wirtschaftsinformatik", "Wirtschaftsingenieurwesen",
    "Betriebswirtschaftslehre", "E-Commerce", "Data Science & Artificial Intelligence",
    "IT-Sicherheit", "Sustainable & Digital Business Management", "Computer Games Technology",
    "Smart Technology", "IT-Management, Consulting (und Auditing)"
]
DOCTYPES = ["Modulhandbuch", "Studien- und Prüfungsordnung", "Modulübersicht", "Studienverlaufsplan"]
STATUSES = ["aktuell", "archiviert"]


# -----------------------------------------
# INTENT & CONTEXT DETECTION
# -----------------------------------------
def detect_context(query, context):
    """
    Hybrid intent detection using:
      - LLM classification for FH Wedel study topics
      - Fallback keyword rules
      - Merges multi-value lists safely
    """
    q_low = query.lower()
    new_context = {}

    # --- Step 1️⃣ LLM-based detection ---
    prompt = f"""
    Du bist ein Klassifizierungsassistent für die Fachhochschule Wedel.

    Analysiere die folgende Anfrage eines Studierenden und gib passende Kategorien zurück.
    Mehrfachauswahlen sind erlaubt, wenn die Anfrage mehrere Bereiche betrifft
    (z. B. „alle Master-Studiengänge“ → ["Master"], „BWL und Wirtschaftsinformatik“ → ["Betriebswirtschaftslehre", "Wirtschaftsinformatik"]).

    **Mögliche Werte:**
    - degree: {DEGREES}
    - program: {PROGRAMS}
    - doctype: {DOCTYPES}
    - status: {STATUSES}

    **Beispielausgabe (im JSON-Format):**
    {{
        "degree": ["Bachelor", "Master"],
        "program": ["Informatik", "Wirtschaftsinformatik"],
        "doctype": ["Modulhandbuch"],
        "status": ["aktuell"]
    }}

    Wenn keine Zuordnung möglich ist, gib eine leere Liste [] zurück.

    Anfrage: "{query}"
    """

    try:
        resp = client.chat.completions.create(
            model=CLASSIFY_MODEL,
            messages=[
                {"role": "system", "content": "Du bist ein präziser Klassifizierer für Studienanfragen der FH Wedel."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        content = resp.choices[0].message.content.strip()
        parsed = json.loads(content)

        if isinstance(parsed, dict):
            for key in ["degree", "program", "doctype", "status"]:
                val = parsed.get(key)
                if isinstance(val, list):
                    new_context[key] = val
                elif val:
                    new_context[key] = [val]
    except Exception as e:
        print(f"⚠️ Intent LLM failed: {e}")

    # --- Step 2️⃣ Heuristic fallback ---
    if not new_context.get("degree"):
        if "master" in q_low:
            new_context["degree"] = ["Master"]
        elif "bachelor" in q_low:
            new_context["degree"] = ["Bachelor"]

    if not new_context.get("program"):
        prog_map = {
            "informatik": "Informatik",
            "wirtschaftsinformatik": "Wirtschaftsinformatik",
            "wirtschaftsingenieur": "Wirtschaftsingenieurwesen",
            "bwl": "Betriebswirtschaftslehre",
            "data science": "Data Science & Artificial Intelligence",
            "e-commerce": "E-Commerce",
            "it-sicherheit": "IT-Sicherheit",
            "consulting": "IT-Management, Consulting (und Auditing)",
            "games": "Computer Games Technology",
            "smart tech": "Smart Technology",
        }
        for key, val in prog_map.items():
            if key in q_low:
                new_context["program"] = [val]
                break

    if not new_context.get("doctype"):
        doc_map = {
            "modulhandbuch": "Modulhandbuch",
            "studienordnung": "Studien- und Prüfungsordnung",
            "prüfungsordnung": "Studien- und Prüfungsordnung",
            "spo": "Studien- und Prüfungsordnung",
            "curriculum": "Studienverlaufsplan",
            "modulübersicht": "Modulübersicht",
        }
        for key, val in doc_map.items():
            if key in q_low:
                new_context["doctype"] = [val]
                break

    if not new_context.get("status"):
        if any(word in q_low for word in ["alt", "früher", "veraltet", "vorherige"]):
            new_context["status"] = ["archiviert"]
        else:
            new_context["status"] = ["aktuell"]

    # --- Step 3️⃣ Merge with persistent context ---
    for key in ["degree", "program", "doctype", "status"]:
        if key in new_context:
            old_vals = context.get(key)
            if isinstance(old_vals, str):
                old_vals = [old_vals]
            elif not isinstance(old_vals, list):
                old_vals = []

            new_vals = new_context[key]
            if isinstance(new_vals, str):
                new_vals = [new_vals]

            merged = list(sorted(set(old_vals + new_vals)))
            context[key] = merged

    return context


# -----------------------------------------
# UTILITIES
# -----------------------------------------
def format_context(context_state):
    """Format context info for clean display in the chat."""
    parts = []
    for key in ["degree", "program", "doctype", "status"]:
        val = context_state.get(key)
        if not val:
            continue
        if isinstance(val, list):
            val_str = ", ".join(val)
        else:
            val_str = str(val)
        parts.append(f"{key}={val_str}")
    return " | ".join(parts)


def build_prompt(query, docs, memory_summary, memory_buffer_text):
    context_docs = "\n\n---\n\n".join(docs)

    return (
        "You are a helpful FH-Wedel assistant.\n"
        "Answer strictly based on either the RAG context or chat memory.\n"
        "If unsure, say 'I don't know'.\n\n"

        f"### Conversation Memory (summary)\n{memory_summary or 'None'}\n\n"
        f"### Recent Conversation\n{memory_buffer_text or 'None'}\n\n"
        f"### RAG Documents\n{context_docs}\n\n"
        f"### Question\n{query}\n\n"
        "### Answer in the same language as the question"
    )


def update_memory_summary(memory_summary, conversation_buffer):
    if not conversation_buffer:
        return memory_summary

    convo_text = "\n".join(
        f"{role}: {msg}" for role, msg in conversation_buffer
    )

    prompt = f"""
    Condense the conversation below into a short memory summary.
    Preserve important facts, user goals, preferences, and constraints.
    Exclude small talk, greetings, and irrelevant phrasing.

    Existing summary:
    {memory_summary}

    Conversation to add:
    {convo_text}

    Updated memory summary:
    """

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "system", "content": prompt}],
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return memory_summary

# -----------------------------------------
# MAIN CHAT LOOP
# -----------------------------------------
def chat_loop():
    print("🧠 FH-Wedel Chatbot ready! Type 'exit' to quit.\n")

    context_state = {"degree": [], "program": [], "doctype": [], "status": ["aktuell"]}

    while True:
        query = input("👤 You: ").strip()
        if not query:
            continue
        if query.lower() in ["exit", "quit", "q"]:
            break

        # Handle meta-commands
        if query.lower() == "show context":
            print(f"🧩 Active context: {format_context(context_state) or 'none'}\n")
            continue
        if query.lower() == "clear context":
            context_state = {"degree": [], "program": [], "doctype": [], "status": ["aktuell"]}
            print("🧹 Context cleared (status=aktuell retained)\n")
            continue

        # --- Detect and update context ---
        context_state = detect_context(query, context_state)

        # --- Show active filters ---
        print(f"🧩 Active context: {format_context(context_state) or 'none'}")

        # --- Retrieve context-relevant chunks ---
        chunks = retrieve_relevant_chunks(
            query,
            degree=context_state.get("degree"),
            program=context_state.get("program"),
            doctype=context_state.get("doctype"),
            status=context_state.get("status")
        )

        if not chunks:
            print("\n🤖 Assistant:\nI don't know.\n" + "=" * 100 + "\n")
            continue

        # --- Build and send chat completion ---
        # build previous messages text
        memory_buffer_text = "\n".join(f"{role}: {msg}" for role, msg in conversation_buffer)

        prompt = build_prompt(query, chunks, memory_summary, memory_buffer_text)

        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        answer = completion.choices[0].message.content
        # store chat turn
        conversation_buffer.append(("user", query))
        conversation_buffer.append(("assistant", answer))

        # summarize if buffer full
        if len(conversation_buffer) == MAX_MEMORY_TURNS:
            memory_summary = update_memory_summary(memory_summary, conversation_buffer)
            conversation_buffer.clear()

        print("\n🤖 Assistant:\n" + textwrap.fill(answer, width=100))
        print("\n" + "=" * 100 + "\n")


if __name__ == "__main__":
    chat_loop()
