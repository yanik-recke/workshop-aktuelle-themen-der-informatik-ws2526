import json
from typing import Dict, List

from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret

from config import (
    DEGREES,
    PROGRAMS,
    DOCTYPES,
    STATUSES,
    OPENAI_CLASSIFIER_MODEL,
)


def _is_broad_query(q: str) -> bool:
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


def _get_classifier() -> OpenAIGenerator:
    # eigener Client für Klassifizierung
    return OpenAIGenerator(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        model=OPENAI_CLASSIFIER_MODEL,
        generation_kwargs={"temperature": 0},
    )


def detect_context(query: str, context: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Kontext-Erkennung wie in deinem alten Projekt:
    - LLM-basiert (konservativ, JSON-only)
    - plus einfache Keyword-Fallbacks
    - Merge-Strategie: degree/status union, program/doctype override
    """
    q_low = query.lower()
    new_context: Dict[str, List[str]] = {}
    classifier = _get_classifier()

    # ---------- 1) LLM-Klassifikation ----------
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
- doctype: {list(DOCTYPES.keys())}
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
        resp = classifier.run(prompt)
        raw = resp["replies"][0].strip()
        parsed = json.loads(raw)

        for key in ["degree", "program", "doctype", "status"]:
            val = parsed.get(key, [])
            if isinstance(val, str):
                val = [val]
            if not isinstance(val, list):
                continue

            allowed_map = {
                "degree": DEGREES,
                "program": PROGRAMS,
                "doctype": list(DOCTYPES.keys()),
                "status": STATUSES,
            }
            allowed = set(allowed_map[key])
            cleaned = [v for v in val if v in allowed]
            new_context[key] = cleaned
    except Exception as e:
        print(f"⚠️ Intent LLM failed: {e}")

    # ---------- 2) Heuristische Fallbacks ----------
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

    # ---------- 3) Merge-Logik ----------
    broad = _is_broad_query(query)

    # degree + status: union
    for key in ["degree", "status"]:
        vals = new_context.get(key)
        if vals:
            old = context.get(key, [])
            if isinstance(old, str):
                old = [old]
            merged = sorted(set(old + vals))
            context[key] = merged

    # program + doctype: override, bei breiten Fragen direkt die erkannten Werte verwenden
    for key in ["program", "doctype"]:
        vals = new_context.get(key)
        if not vals:
            continue
        if broad:
            context[key] = vals
        else:
            context[key] = vals

    return context
