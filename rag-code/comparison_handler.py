"""
Comparison query handler for multi-entity retrieval.
Detects comparison questions and retrieves documents for each entity separately.
"""
import re
from typing import List, Dict, Tuple, Optional
from haystack.dataclasses import Document

from hybrid_retrieval import hybrid_search


# Patterns that indicate comparison questions
COMPARISON_PATTERNS = [
    r"unterschied(?:e|en)?\s+zwischen\s+(.+?)\s+und\s+(.+?)(?:\?|$|\.)",
    r"vergleich(?:e|en)?\s+(.+?)\s+(?:mit|und)\s+(.+?)(?:\?|$|\.)",
    r"was\s+ist\s+(?:der\s+)?unterschied\s+zwischen\s+(.+?)\s+und\s+(.+?)(?:\?|$|\.)",
    r"(.+?)\s+(?:vs\.?|versus|oder)\s+(.+?)(?:\?|$|\.)",
    r"difference\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$|\.)",
    r"compare\s+(.+?)\s+(?:with|and|to)\s+(.+?)(?:\?|$|\.)",
]


def detect_comparison_query(query: str) -> Optional[Tuple[str, str]]:
    """
    Detect if query is a comparison and extract the two entities.
    Returns (entity1, entity2) if comparison detected, None otherwise.
    """
    query_lower = query.lower()
    
    for pattern in COMPARISON_PATTERNS:
        match = re.search(pattern, query_lower, re.IGNORECASE)
        if match:
            entity1 = match.group(1).strip()
            entity2 = match.group(2).strip()
            # Clean up common artifacts
            entity1 = re.sub(r'^(dem?|der|die|das)\s+', '', entity1)
            entity2 = re.sub(r'^(dem?|der|die|das)\s+', '', entity2)
            return (entity1, entity2)
    
    return None


def extract_keywords_for_entity(entity: str) -> List[str]:
    """Extract searchable keywords from an entity name."""
    # Remove common filler words
    stopwords = {'dem', 'der', 'die', 'das', 'von', 'aus', 'und', 'im', 'am'}
    words = entity.split()
    keywords = [w.lower() for w in words if w.lower() not in stopwords and len(w) > 2]
    return keywords


def retrieve_for_comparison(
    query: str,
    entity1: str,
    entity2: str,
    top_k_per_entity: int = 3,
    expanded_query: str = None,
) -> Tuple[List[Document], List[Document]]:
    """
    Retrieve documents for each entity in a comparison.
    Returns two lists: (docs_for_entity1, docs_for_entity2)
    """
    # Extract keywords for each entity
    keywords1 = extract_keywords_for_entity(entity1)
    keywords2 = extract_keywords_for_entity(entity2)
    
    print(f"  Comparing: '{entity1}' vs '{entity2}'")
    print(f"  Keywords 1: {keywords1}")
    print(f"  Keywords 2: {keywords2}")
    
    # Retrieve for each entity
    docs1 = hybrid_search(
        query=f"{entity1} FH Wedel",
        top_k=top_k_per_entity,
        keywords=keywords1,
    )
    
    docs2 = hybrid_search(
        query=f"{entity2} FH Wedel",
        top_k=top_k_per_entity,
        keywords=keywords2,
    )
    
    return docs1, docs2


def merge_comparison_docs(
    docs1: List[Document],
    docs2: List[Document],
    entity1: str,
    entity2: str,
) -> List[Document]:
    """
    Merge documents from both entities, tagging them with their source.
    """
    merged = []
    
    # Tag docs with their entity source
    for doc in docs1:
        doc.meta = doc.meta or {}
        doc.meta["comparison_entity"] = entity1
        merged.append(doc)
    
    for doc in docs2:
        doc.meta = doc.meta or {}
        doc.meta["comparison_entity"] = entity2
        merged.append(doc)
    
    return merged


def handle_comparison_query(
    query: str,
    expanded_query: str = None,
    top_k: int = 5,
) -> Optional[Tuple[List[Document], str, str]]:
    """
    Handle a comparison query by detecting entities and retrieving for each.
    
    Returns:
        Tuple of (merged_docs, entity1, entity2) if comparison detected
        None if not a comparison query
    """
    entities = detect_comparison_query(query)
    
    if not entities:
        return None
    
    entity1, entity2 = entities
    print(f"Comparison detected: '{entity1}' vs '{entity2}'")
    
    # Get documents for each entity
    top_k_per = max(2, top_k // 2)  # Split top_k between entities
    docs1, docs2 = retrieve_for_comparison(
        query=query,
        entity1=entity1,
        entity2=entity2,
        top_k_per_entity=top_k_per,
        expanded_query=expanded_query,
    )
    
    # Merge with entity tags
    merged = merge_comparison_docs(docs1, docs2, entity1, entity2)
    
    return merged, entity1, entity2
