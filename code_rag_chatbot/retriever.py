from qdrant_client import QdrantClient
from openai import OpenAI
from config import *
from utils import sanitize_key

client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL)

def retrieve_relevant_chunks(
    query,
    top_k=25,
    degree=None,
    program=None,
    doctype=None,
    status=None
):
    """
    Retrieve top_k semantically relevant chunks from Qdrant.
    Unterstützt Mehrfachwerte (Listen) für degree/program/doctype/status.
    Filter werden als plain-JSON gebaut, um Versions-/Unicode-Probleme zu vermeiden.
    """

    # 1) Embed Query
    emb = client.embeddings.create(input=query, model=EMBED_MODEL).data[0].embedding

    # 2) Query-Filter als dict (REST-Form) konstruieren
    def to_list(v):
        if not v:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)

    def field_eq(path):
        # FieldCondition in REST-Form
        return {"key": path, "match": {"value": 1}}

    must = []

    # Innerhalb einer Kategorie: OR (should). Zwischen Kategorien: AND (must).
    for prefix, values in [
        ("degree_onehot",  to_list(degree)),
        ("program_onehot", to_list(program)),
        ("doctype_onehot", to_list(doctype)),
        ("status_onehot",  to_list(status)),
    ]:
        if not values:
            continue

        # Pfad = "<prefix>.<wert>" — Achtung: exakt wie in der Payload indexiert
        should = [field_eq(f"{prefix}.{sanitize_key(v)}") for v in values]

        if len(should) == 1:
            must.append(should[0])            # nur eine Bedingung → direkt in must
        else:
            must.append({"should": should})   # mehrere Werte → OR-Gruppe

    query_filter = {"must": must} if must else None
    print("DEBUG QUERY_FILTER: \n", query_filter)
    # 3) Suche
    # Achtung: Bei älteren qdrant-client Versionen heißt der Parameter 'query_filter'
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=emb,
        query_filter=query_filter, 
        limit=top_k,
        with_payload=True
    )

    # 4) Deduplizieren
    docs, seen = [], set()
    for r in results:
        text = (r.payload.get("text") or "").strip()
        if not text or text in seen:
            continue
        combos = r.payload.get("combos", [])
        meta_info = "; ".join([" / ".join([c for c in combo if c]) for combo in combos])
        docs.append(f"📘 {meta_info}\n{text}")
        seen.add(text)

    return docs
