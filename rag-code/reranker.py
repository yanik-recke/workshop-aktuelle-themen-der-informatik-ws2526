"""
Chunk reranker for improving retrieval quality.
Supports multiple methods: keyword-based (fast) and LLM-based (slow but accurate).
"""
import math
import re
from collections import Counter
from typing import List, Tuple, Dict
from haystack.dataclasses import Document
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret

from config import OPENAI_RERANK_MODEL


# German stopwords to ignore in scoring
GERMAN_STOPWORDS = {
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer", "eines",
    "und", "oder", "aber", "wenn", "als", "ob", "dass", "weil",
    "ich", "du", "er", "sie", "es", "wir", "ihr", "mich", "dich", "sich",
    "ist", "sind", "war", "waren", "bin", "bist", "haben", "hat", "hatte",
    "wird", "werden", "wurde", "kann", "können", "muss", "müssen", "soll",
    "was", "wer", "wie", "wo", "wann", "warum", "welche", "welcher",
    "für", "mit", "bei", "von", "aus", "nach", "zu", "über", "unter",
    "auf", "an", "in", "im", "am", "nicht", "kein", "keine",
    "auch", "noch", "schon", "nur", "sehr", "mehr", "viel",
}


def tokenize(text: str) -> List[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric, filter stopwords."""
    words = re.findall(r'\b\w+\b', text.lower())
    return [w for w in words if w not in GERMAN_STOPWORDS and len(w) > 2]


def bm25_score(query_terms: List[str], doc_terms: List[str], avg_doc_len: float, k1: float = 1.5, b: float = 0.75) -> float:
    """
    BM25-style scoring for a single document.
    Fast keyword-based relevance scoring.
    """
    if not query_terms or not doc_terms:
        return 0.0
    
    doc_len = len(doc_terms)
    doc_freq = Counter(doc_terms)
    
    score = 0.0
    for term in query_terms:
        if term in doc_freq:
            tf = doc_freq[term]
            # Simplified BM25 (without IDF since we score single docs)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
            score += numerator / denominator
    
    return score


def keyword_overlap_score(query_terms: List[str], doc_terms: List[str]) -> float:
    """
    Simple keyword overlap scoring.
    Returns percentage of query terms found in document.
    """
    if not query_terms:
        return 0.0
    
    doc_set = set(doc_terms)
    matches = sum(1 for t in query_terms if t in doc_set)
    return matches / len(query_terms)


def exact_phrase_bonus(query: str, doc_content: str) -> float:
    """Bonus for exact phrase matches (case-insensitive)."""
    query_lower = query.lower()
    doc_lower = doc_content.lower()
    
    # Check for exact query match
    if query_lower in doc_lower:
        return 0.3
    
    # Check for significant subphrases (3+ words)
    words = query_lower.split()
    if len(words) >= 3:
        for i in range(len(words) - 2):
            phrase = " ".join(words[i:i+3])
            if phrase in doc_lower:
                return 0.15
    
    return 0.0


def rerank_keyword_based(
    query: str,
    documents: List[Document],
    top_k: int = 5,
) -> List[Document]:
    """
    Fast keyword-based reranking using BM25-style scoring.
    No LLM calls - very fast!
    """
    if not documents:
        return []
    
    query_terms = tokenize(query)
    if not query_terms:
        return documents[:top_k]
    
    # Tokenize all documents
    doc_data = []
    total_len = 0
    for doc in documents:
        terms = tokenize(doc.content)
        doc_data.append((doc, terms))
        total_len += len(terms)
    
    avg_doc_len = total_len / len(documents) if documents else 1
    
    # Score each document
    scored_docs: List[Tuple[Document, float]] = []
    for doc, terms in doc_data:
        # Combine multiple scoring methods
        bm25 = bm25_score(query_terms, terms, avg_doc_len)
        overlap = keyword_overlap_score(query_terms, terms)
        phrase_bonus = exact_phrase_bonus(query, doc.content)
        
        # Weighted combination
        score = (bm25 * 0.5) + (overlap * 0.3) + (phrase_bonus * 0.2)
        
        # Keep original semantic score as a factor
        if doc.score is not None:
            score = score * 0.6 + doc.score * 0.4
        
        scored_docs.append((doc, score))
    
    # Sort by score
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Update scores and return
    results = []
    for doc, score in scored_docs[:top_k]:
        doc.score = score
        results.append(doc)
    
    return results


BATCH_RERANK_PROMPT = """Du bist ein Experte für Relevanz-Bewertung.

Frage: {query}

Hier sind {num_chunks} Textabschnitte. Sortiere sie nach Relevanz für die Frage.
Gib NUR die Chunk-Nummern zurück, sortiert vom relevantesten zum unwichtigsten.
Format: Eine Zeile mit kommagetrennten Nummern, z.B.: 3,1,5,2,4

{chunks}

Sortierte Reihenfolge (nur Nummern):"""


def rerank_chunks(
    query: str,
    documents: List[Document],
    top_k: int = 5,
) -> List[Document]:
    """
    Rerank documents using LLM in a SINGLE API call.
    All chunks are sent together and the LLM returns the ranked order.
    
    Much faster than per-chunk scoring!
    """
    if not documents:
        return []
    
    if len(documents) <= 2:
        return documents
    
    # Build chunks text with numbered labels
    chunks_text = []
    for i, doc in enumerate(documents, 1):
        preview = doc.content[:800] if len(doc.content) > 800 else doc.content
        preview = preview.replace('\n', ' ').strip()
        chunks_text.append(f"[Chunk {i}]: {preview}")
    
    prompt = BATCH_RERANK_PROMPT.format(
        query=query,
        num_chunks=len(documents),
        chunks="\n\n".join(chunks_text)
    )
    
    generator = OpenAIGenerator(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        api_base_url=None,  # Force using OpenAI API
        model=OPENAI_RERANK_MODEL,
        timeout=60.0,
        generation_kwargs={"temperature": 0.0, "max_tokens": 100},
    )
    
    try:
        result = generator.run(prompt=prompt)
        reply = result.get("replies", [""])[0].strip()
        
        # Parse the ranked order from response
        ranked_indices = parse_ranked_order(reply, len(documents))
        
        # Reorder documents based on LLM ranking
        reranked = []
        for idx in ranked_indices[:top_k]:
            doc = documents[idx]
            doc.score = 1.0 - (len(reranked) / len(ranked_indices))  # Score by position
            reranked.append(doc)
        
        return reranked
        
    except Exception as e:
        print(f"[Reranker] LLM batch rerank failed: {e}")
        return documents[:top_k]


def parse_ranked_order(reply: str, num_docs: int) -> List[int]:
    """Parse LLM response to extract ranked chunk indices (0-based)."""
    # Extract numbers from response
    numbers = re.findall(r'\d+', reply)
    
    seen = set()
    indices = []
    for num_str in numbers:
        try:
            num = int(num_str)
            if 1 <= num <= num_docs and num not in seen:
                indices.append(num - 1)  # Convert to 0-based
                seen.add(num)
        except:
            pass
    
    # Add any missing indices at the end
    for i in range(num_docs):
        if i not in seen:
            indices.append(i)
    
    return indices


def rerank_with_metadata_boost(
    query: str,
    documents: List[Document],
    top_k: int = 5,
    use_llm_rerank: bool = False,  # Default to fast keyword-based
) -> List[Document]:
    """
    Rerank documents with multiple strategies:
    
    1. Keyword-based reranking (BM25 + overlap) - FAST, always applied
    2. LLM reranking (optional) - SLOW but more accurate
    3. Metadata boosts (program match, doctype match)
    
    Args:
        use_llm_rerank: If True, uses slow LLM-based scoring. Default False for speed.
    """
    if not documents:
        return []
    
    # First, apply keyword-based reranking (fast)
    if len(documents) > 2:
        documents = rerank_keyword_based(query, documents, top_k=min(len(documents), top_k * 2))
    
    # Optionally apply LLM reranking (slow but accurate)
    if use_llm_rerank and len(documents) > 2:
        documents = rerank_chunks(query, documents, top_k=min(len(documents), top_k * 2))
    
    # Then apply metadata boosts
    query_lower = query.lower()
    
    for doc in documents:
        meta = doc.meta or {}
        boost = 0.0
        
        # Boost if program name appears in query
        program = (meta.get("program") or "").lower()
        if program and program in query_lower:
            boost += 0.1
        
        # Boost SPO docs for regulation questions
        filename = (meta.get("filename") or "").lower()
        if any(kw in query_lower for kw in ["regelstudienzeit", "prüfung", "zulassung", "semester"]):
            if "spo" in filename or "pvo" in filename:
                boost += 0.15
        
        # Boost Modulhandbuch for module questions
        if any(kw in query_lower for kw in ["modul", "vorlesung", "inhalt", "lernziel"]):
            if "modulhandbuch" in filename:
                boost += 0.15
        
        doc.score = (doc.score or 0.5) + boost
    
    # Re-sort after boosts
    documents.sort(key=lambda d: d.score or 0, reverse=True)
    
    return documents[:top_k]
