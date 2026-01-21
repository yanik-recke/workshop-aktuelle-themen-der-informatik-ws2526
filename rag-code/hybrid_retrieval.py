"""
Hybrid retrieval combining keyword search with semantic search.
This helps find documents when the embedding model misses exact keyword matches.
"""
from typing import List, Dict, Optional
from haystack.dataclasses import Document
from haystack.components.embedders import OpenAITextEmbedder
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack.utils import Secret
from qdrant_client import models

from config import OPENAI_EMBED_MODEL, OPENAI_EMBED_BASE_URL, TOP_K, QDRANT_COLLECTION
from document_store import get_document_store


def keyword_search(query: str, top_k: int = 10, keywords: List[str] = None) -> List[Document]:
    """
    Search for documents containing query keywords using Qdrant scroll + filter.
    This is a simple keyword match, not BM25, but catches exact matches the embedding might miss.
    
    Args:
        query: The query string (used if keywords not provided)
        keywords: Pre-extracted keywords from query expansion (preferred)
    """
    store = get_document_store()
    
    # Use provided keywords or extract from query
    if keywords:
        search_keywords = [k.lower() for k in keywords if len(k) > 2]
    else:
        search_keywords = [w.lower() for w in query.split() if len(w) > 3]
    
    keywords = search_keywords
    
    if not keywords:
        return []
    
    # Get all documents and filter by keyword (simple approach for local Qdrant)
    # For production, you'd use Qdrant's full-text search or a separate BM25 index
    all_docs = store.filter_documents()
    
    matched_docs = []
    for doc in all_docs:
        content_lower = doc.content.lower()
        # Score by how many keywords match
        matches = sum(1 for kw in keywords if kw in content_lower)
        if matches > 0:
            # Create a pseudo-score based on keyword matches
            doc.score = matches / len(keywords)
            matched_docs.append(doc)
    
    # Sort by score (most keyword matches first)
    matched_docs.sort(key=lambda d: d.score, reverse=True)
    
    return matched_docs[:top_k]


def semantic_search(query: str, top_k: int = 10, filters: Dict = None) -> List[Document]:
    """
    Standard semantic search using embeddings.
    """
    store = get_document_store()
    
    embedder = OpenAITextEmbedder(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        api_base_url=OPENAI_EMBED_BASE_URL,  # Use local Ollama for embeddings
        model=OPENAI_EMBED_MODEL,
    )
    
    retriever = QdrantEmbeddingRetriever(
        document_store=store,
        top_k=top_k,
        return_embedding=False,
    )
    
    # Embed the query
    emb_result = embedder.run(text=query)
    query_embedding = emb_result["embedding"]
    
    # Retrieve
    if filters:
        ret_result = retriever.run(query_embedding=query_embedding, filters=filters)
    else:
        ret_result = retriever.run(query_embedding=query_embedding)
    
    return ret_result.get("documents", [])


# Known program names for exact matching
KNOWN_PROGRAMS = [
    "informatik", "medieninformatik", "technische informatik", "wirtschaftsinformatik",
    "e-commerce", "betriebswirtschaftslehre", "bwl", "it-ingenieurwesen",
    "computer games technology", "data science", "it-management", "it-sicherheit",
    "smart technology", "wirtschaftsingenieurwesen",
]

# Query patterns that indicate which doctype is most relevant
REGULATION_KEYWORDS = [
    "regelstudienzeit", "prüfung", "klausur", "anmeldung", "abmeldung", "versuch",
    "wiederholung", "freiversuch", "note", "bewertung", "zulassung", "voraussetzung",
    "anerkennung", "urlaubssemester", "exmatrikulation", "immatrikulation",
    "prüfungsordnung", "studienordnung", "regelung", "frist", "antrag", "ausland",
    "praktikum", "thesis", "abschlussarbeit", "bachelor", "master", "ects",
    "creditpoints", "credit", "leistungspunkte", "duales studium", "dual",
]

MODULE_KEYWORDS = [
    "modul", "module", "vorlesung", "lernziel", "inhalt", "dozent", "professor",
    "lehrveranstaltung", "seminar", "praktikum", "labor", "projekt",
    "welche fächer", "welche module", "was lernt", "curriculum", "studienverlauf",
    "semester", "stundenplan", "veranstaltung",
]


def detect_query_intent(query: str, keywords: List[str] = None) -> str:
    """
    Detect what type of document would best answer this query.
    Returns: 'regulation', 'module', or 'general'
    """
    query_lower = query.lower()
    all_terms = query_lower + " " + " ".join(keywords or [])
    
    regulation_score = sum(1 for kw in REGULATION_KEYWORDS if kw in all_terms)
    module_score = sum(1 for kw in MODULE_KEYWORDS if kw in all_terms)
    
    if regulation_score > module_score and regulation_score > 0:
        return "regulation"
    elif module_score > regulation_score and module_score > 0:
        return "module"
    return "general"


def extract_program_from_query(query: str, keywords: List[str] = None) -> Optional[str]:
    """
    Try to extract a specific program name from the query.
    Returns the program name if found, None otherwise.
    """
    query_lower = query.lower()
    search_terms = keywords or []
    
    # Check for exact program matches in query
    for prog in KNOWN_PROGRAMS:
        if prog in query_lower:
            return prog
    
    # Check keywords
    for kw in search_terms:
        kw_lower = kw.lower()
        for prog in KNOWN_PROGRAMS:
            if prog == kw_lower:
                return prog
    
    return None


def hybrid_search(
    query: str,
    top_k: int = TOP_K,
    filters: Dict = None,
    keyword_weight: float = 0.5,
    semantic_weight: float = 0.5,
    original_query: str = None,
    keywords: List[str] = None,
) -> List[Document]:
    """
    Combine keyword and semantic search results.
    
    Strategy:
    1. Run keyword search using ENHANCED KEYWORDS (spelling-corrected, with synonyms)
    2. Run semantic search on EXPANDED query for conceptual matches
    3. Merge results, boosting docs that appear in both
    4. Boost exact program matches (e.g., "Informatik" > "Technische Informatik")
    5. Return top_k unique documents
    
    Args:
        query: The (possibly expanded) query for semantic search
        original_query: The original user query (for display)
        keywords: Enhanced keywords from query expansion (preferred for keyword search)
    """
    # Get more results from each to allow for merging
    fetch_k = top_k * 3
    
    # Try to extract a specific program the user is asking about
    target_program = extract_program_from_query(original_query or query, keywords)
    
    # Use provided keywords for keyword search, fallback to query extraction
    keyword_docs = keyword_search(query, top_k=fetch_k, keywords=keywords)
    semantic_docs = semantic_search(query, top_k=fetch_k, filters=filters)
    
    # Build a map of doc_id -> combined score
    doc_scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}
    
    # Score keyword results
    for i, doc in enumerate(keyword_docs):
        doc_id = doc.id or str(hash(doc.content[:100]))
        # Normalize rank to score (higher rank = higher score)
        rank_score = 1.0 - (i / max(len(keyword_docs), 1))
        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + keyword_weight * rank_score
        doc_map[doc_id] = doc
    
    # Score semantic results
    for i, doc in enumerate(semantic_docs):
        doc_id = doc.id or str(hash(doc.content[:100]))
        # Use actual similarity score if available, otherwise use rank
        if doc.score is not None:
            score = doc.score
        else:
            score = 1.0 - (i / max(len(semantic_docs), 1))
        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + semantic_weight * score
        doc_map[doc_id] = doc
    
    # Detect query intent to boost relevant document types
    query_intent = detect_query_intent(original_query or query, keywords)
    
    # Apply context-aware scoring
    for doc_id, doc in doc_map.items():
        meta = doc.meta or {}
        doc_program = (meta.get("program") or "").lower()
        doc_doctype = (meta.get("doctype") or "").lower()
        doc_filename = (meta.get("filename") or "").lower()
        
        # 1. Boost exact program matches (reduced from 0.5 to 0.15)
        if target_program:
            if doc_program == target_program:
                doc_scores[doc_id] += 0.15  # Moderate boost for exact match
            elif target_program in doc_program and doc_program != target_program:
                doc_scores[doc_id] -= 0.05  # Small penalty for partial match
        
        # 2. Boost based on query intent and document type
        if query_intent == "regulation":
            # Boost SPO, PVO, ZLO for regulation questions
            if any(x in doc_filename for x in ["spo", "pvo", "zlo", "richtlinien"]):
                doc_scores[doc_id] += 0.25
            elif "modulhandbuch" in doc_filename:
                doc_scores[doc_id] += 0.1  # Modulhandbuch also has some regulation info
        elif query_intent == "module":
            # Boost curriculum and modulhandbuch for module questions
            if "curriculum" in doc_filename:
                doc_scores[doc_id] += 0.2
            elif "modulhandbuch" in doc_filename:
                doc_scores[doc_id] += 0.25  # Modulhandbuch is best for module details
            elif "moduluebersicht" in doc_filename:
                doc_scores[doc_id] += 0.15
    
    # Sort by combined score
    sorted_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
    
    # Return top_k with updated scores
    results = []
    for doc_id in sorted_ids[:top_k]:
        doc = doc_map[doc_id]
        doc.score = doc_scores[doc_id]
        results.append(doc)
    
    return results


def print_retrieval_debug(docs: List[Document], query: str):
    """Print debug info about retrieved documents."""
    print(f"\n🔎 Retrieved {len(docs)} documents for: '{query}'")
    for idx, d in enumerate(docs, start=1):
        meta = d.meta or {}
        score = d.score
        title_parts = [
            str(meta.get("doctype") or ""),
            str(meta.get("program") or ""),
            str(meta.get("degree") or ""),
        ]
        title = " | ".join([p for p in title_parts if p])
        filename = meta.get("filename") or ""
        
        # Check if query keywords are in content
        query_words = [w.lower() for w in query.split() if len(w) > 3]
        content_lower = d.content.lower()
        keyword_matches = [w for w in query_words if w in content_lower]
        
        if score is not None:
            print(f"[{idx}] score={score:.4f} | {title} | {filename}")
        else:
            print(f"[{idx}] {title} | {filename}")
        
        if keyword_matches:
            print(f"    ✓ Contains keywords: {keyword_matches}")
        
        preview = (d.content or "")[:500]
        print(preview)
        print("---")
