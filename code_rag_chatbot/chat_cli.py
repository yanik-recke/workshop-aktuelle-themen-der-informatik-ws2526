import json
import textwrap
from collections import deque
from openai import OpenAI
from retriever import retrieve_relevant_chunks
from config import *

# -----------------------------------------
# CLIENT SETUP
# -----------------------------------------
client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
MAX_MEMORY_TURNS = 5
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
def is_broad_query(q: str) -> bool:
    q = q.lower()
    return any(
        kw in q
        for kw in [
            "alle studiengänge",
            "welche studiengänge",
            "übersicht",
            "gesamt",
            "insgesamt",
            "alle module",
        ]
    )


def detect_context(query, context):
    """
    Hybrid intent detection using:
      - LLM classification for FH Wedel study topics (konservativ!)
      - Fallback keyword rules
      - Smarte Merge-Strategie:
          * program/doctype: explizite Angaben überschreiben bisherigen Kontext
          * degree/status: werden zusammengeführt
    """
    q_low = query.lower()
    new_context = {}

    # -------- 1) LLM-basierte Erkennung (konservativ) --------
    prompt = f"""
Du bist ein Klassifizierungsassistent für die Fachhochschule Wedel.

Analysiere die folgende Anfrage eines Studierenden und weise sie, falls eindeutig passend,
zu den folgenden Kategorien zu.

WICHTIG:
- Sei konservativ.
- Gib NUR Werte zurück, die explizit genannt oder eindeutig aus dem Inhalt ableitbar sind.
- Gib KEINE vollständigen Listen aller Studiengänge zurück.
- Rate NICHT. Wenn du unsicher bist, gib [] zurück.

Mögliche Werte:
- degree: {DEGREES}
- program: {PROGRAMS}
- doctype: {DOCTYPES}
- status: {STATUSES}

Erwarte Ausgabe AUSSCHLIESSLICH im JSON-Format, z.B.:
{{
  "degree": ["Bachelor"],
  "program": ["Smart Technology"],
  "doctype": ["Modulhandbuch"],
  "status": ["aktuell"]
}}

Wenn keine sinnvolle Zuordnung möglich ist:
{{
  "degree": [],
  "program": [],
  "doctype": [],
  "status": []
}}

Anfrage: "{query}"
"""

    try:
        resp = client.chat.completions.create(
            model=CLASSIFY_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Du bist ein strenger Klassifizierer. Du gibst nur Werte zurück, die klar im Text liegen. Du spekulierst nicht.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        parsed = json.loads(resp.choices[0].message.content.strip())
        if isinstance(parsed, dict):
            for key in ["degree", "program", "doctype", "status"]:
                val = parsed.get(key, [])
                if isinstance(val, str):
                    val = [val]
                if isinstance(val, list):
                    # nur gültige Werte behalten
                    allowed_map = {
                        "degree": DEGREES,
                        "program": PROGRAMS,
                        "doctype": DOCTYPES,
                        "status": STATUSES,
                    }
                    allowed = set(allowed_map[key])
                    cleaned = [v for v in val if v in allowed]
                    new_context[key] = cleaned
    except Exception as e:
        print(f"⚠️ Intent LLM failed: {e}")

    # -------- 2) Heuristische Fallbacks --------
    if not new_context.get("degree"):
        if "master" in q_low:
            new_context["degree"] = ["Master"]
        elif "bachelor" in q_low:
            new_context["degree"] = ["Bachelor"]

    if not new_context.get("program"):
        prog_map = {
            "smart technology": "Smart Technology",
            "wirtschaftsinformatik": "Wirtschaftsinformatik",
            "informatik": "Informatik",
            "wirtschaftsingenieur": "Wirtschaftsingenieurwesen",
            "bwl": "Betriebswirtschaftslehre",
            "data science": "Data Science & Artificial Intelligence",
            "e-commerce": "E-Commerce",
            "it-sicherheit": "IT-Sicherheit",
            "consulting": "IT-Management, Consulting (und Auditing)",
            "games": "Computer Games Technology",
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
        if any(w in q_low for w in ["alt", "früher", "veraltet", "vorherige"]):
            new_context["status"] = ["archiviert"]
        else:
            new_context["status"] = ["aktuell"]

    # -------- 3) Merge-Logik --------
    broad = is_broad_query(query)

    # degree + status: union (relativ harmlos)
    for key in ["degree", "status"]:
        vals = new_context.get(key)
        if vals:
            old = context.get(key, [])
            if isinstance(old, str):
                old = [old]
            merged = sorted(set(old + vals))
            context[key] = merged

    # program + doctype:
    # - bei expliziten Angaben (z.B. "Smart Technology") → überschreiben
    # - bei breiten Fragen (z.B. "Welche Studiengänge gibt es?") → ersetzen durch new_context
    # - kein blindes Aufsummieren mehr
    for key in ["program", "doctype"]:
        vals = new_context.get(key)
        if not vals:
            continue

        if broad:
            # breite Frage → nutze explizit die erkannten (z.B. mehrere Programme),
            # aber nicht + alte weiterziehen
            context[key] = vals
        else:
            # eher spezifische Frage → override
            context[key] = vals

    return context


# -----------------------------------------
# QUERY EXPANSION
# -----------------------------------------
def expand_query_with_llm(query, memory_summary, conversation_buffer):
    """
    Semantische Query-Erweiterung:
    - Klärt Intention
    - Keine Kategorien, keine künstlichen Listen
    - Keine neuen Fakten
    """
    recent_history = "\n".join(
        f"{role}: {msg}" for role, msg in list(conversation_buffer)[-4:]
    )

    prompt = f"""
You expand user questions into clearer retrieval queries.

Rules:
- Do NOT answer the question.
- Do NOT add degrees, programs, lists, examples or assumptions that were not mentioned.
- Only clarify what the user is asking for (intent, scope, entities).
- Keep the same language as the original question.
- Output must stay short, precise and retrieval-friendly.

Conversation history:
{recent_history or "None"}

Memory summary:
{memory_summary or "None"}

Original user question:
{query}

Return ONLY valid JSON:
{{
  "original_query": "<original question>",
  "expanded_query": "<rewritten query>"
}}
"""

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise query expansion agent. You never invent new facts or lists.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        data = json.loads(resp.choices[0].message.content.strip())
        expanded_query = data.get("expanded_query") or query
        original_query = data.get("original_query") or query

        # kleine Sicherung: kein Roman
        expanded_query = expanded_query.strip()
        if len(expanded_query) > 500:
            expanded_query = query

        print(f"\n🧭 Expanded query for retrieval:\n{expanded_query}\n")
        return expanded_query, original_query
    except Exception as e:
        print(f"⚠️ Query expansion failed: {e}")
        return query, query


# -----------------------------------------
# UTILITIES
# -----------------------------------------
def format_context(context_state):
    parts = []
    for key in ["degree", "program", "doctype", "status"]:
        vals = context_state.get(key)
        if not vals:
            continue
        if isinstance(vals, str):
            vals = [vals]
        parts.append(f"{key}=" + ", ".join(vals))
    return " | ".join(parts)


def build_prompt(expanded_query, docs, memory_summary, memory_buffer_text, original_query):
    context_docs = "\n\n---\n\n".join(docs)
    return (
        "You are a helpful FH-Wedel assistant.\n"
        "Answer strictly based on the provided RAG documents and memory.\n"
        "If the answer is not clearly contained there, say 'I don't know'.\n\n"
        f"### Memory Summary\n{memory_summary or 'None'}\n\n"
        f"### Recent Conversation\n{memory_buffer_text or 'None'}\n\n"
        f"### RAG Documents\n{context_docs}\n\n"
        f"### Original Question\n{original_query}\n\n"
        f"### Expanded Retrieval Query\n{expanded_query}\n\n"
        "### Answer in the same language as the question"
    )


def update_memory_summary(memory_summary, conversation_buffer):
    if not conversation_buffer:
        return memory_summary

    convo_text = "\n".join(f"{role}: {msg}" for role, msg in conversation_buffer)

    prompt = f"""
Condense the conversation below into a short memory summary.
Keep only stable facts, preferences, constraints and goals.
Ignore greetings, filler, and one-off details.

Existing summary:
{memory_summary}

Conversation to add:
{convo_text}

Updated summary:
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
    memory_summary = ""

    while True:
        query = input("👤 You: ").strip()
        if not query:
            continue
        if query.lower() in ["exit", "quit", "q"]:
            break

        if query.lower() == "show context":
            print(f"🧩 Active context: {format_context(context_state) or 'none'}\n")
            continue

        if query.lower() == "clear context":
            context_state = {"degree": [], "program": [], "doctype": [], "status": ["aktuell"]}
            print("🧹 Context cleared (status=aktuell retained)\n")
            continue

        # 1) Kontext erkennen & updaten
        context_state = detect_context(query, context_state)

        # 2) Query neutral erweitern
        expanded_query, original_query = expand_query_with_llm(
            query, memory_summary, conversation_buffer
        )

        print(f"🧩 Active context: {format_context(context_state) or 'none'}")

        # 3) RAG: relevante Chunks holen (auf Basis expanded_query + Filter)
        chunks = retrieve_relevant_chunks(
            expanded_query,
            degree=context_state.get("degree"),
            program=context_state.get("program"),
            doctype=context_state.get("doctype"),
            status=context_state.get("status"),
        )

        if not chunks:
            print("\n🤖 Assistant:\nI don't know.\n" + "=" * 100 + "\n")
            # Turn trotzdem in Memory packen
            conversation_buffer.append(("user", query))
            if len(conversation_buffer) == MAX_MEMORY_TURNS:
                memory_summary = update_memory_summary(memory_summary, conversation_buffer)
                conversation_buffer.clear()
            continue

        # 4) Antwort mit RAG + Memory generieren
        memory_buffer_text = "\n".join(f"{role}: {msg}" for role, msg in conversation_buffer)
        prompt = build_prompt(expanded_query, chunks, memory_summary, memory_buffer_text, original_query)

        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        answer = completion.choices[0].message.content.strip()

        # 5) Verlauf & Memory aktualisieren
        conversation_buffer.append(("user", query))
        conversation_buffer.append(("assistant", answer))

        if len(conversation_buffer) == MAX_MEMORY_TURNS:
            memory_summary = update_memory_summary(memory_summary, conversation_buffer)
            conversation_buffer.clear()

        # 6) Ausgabe
        print("\n🤖 Assistant:\n" + textwrap.fill(answer, width=100))
        print("\n" + "=" * 100 + "\n")


if __name__ == "__main__":
    chat_loop()
