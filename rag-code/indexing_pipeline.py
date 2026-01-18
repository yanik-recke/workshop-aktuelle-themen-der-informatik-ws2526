from __future__ import annotations

import json
import ntpath
import os
from pathlib import Path
from typing import List, Dict

# Set default OPENAI_API_KEY if not set
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "default"

from haystack.dataclasses import Document
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret

from config import OPENAI_EMBED_MODEL, OPENAI_EMBED_BASE_URL
from document_store import get_document_store
from parsers.base_parser import BaseParser
from preprocessing_pipeline import (
    prepare_sources_and_meta,
    normalize_text,
)

from parsers.curriculum_parser import CurriculumParser
from parsers.modulhandbuch_parser import ModulhandbuchParser
from parsers.regulations_parser import RegulationsParser

from metadata_collector import MetadataCollector
import re

PARSERS = [
    CurriculumParser(),
    ModulhandbuchParser(),
    RegulationsParser(),
]


def select_parser(filename: str):
    filename: str = filename.upper()
    if "CURRICULUM" in filename:
        return PARSERS[0]
    if "MODULHANDBUCH" in filename:
        return PARSERS[1]
    if re.match(".*(SPO|PVO|ZLO|RICHTLINIEN).*", filename):
        return PARSERS[2]
    return None

def fallback_extract(path: Path, meta: Dict) -> List[Document]:
    """
    Fallback extractor for markdown files that don't have a specific parser.
    Simply reads the markdown file and creates a single document.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    doc = Document(content=content, meta=meta)
    print(f"[WARN] Fallback-Parser benutzt fuer {path.name}")
    return [doc]



def clean_documents(docs: List[Document]) -> List[Document]:
    cleaner = DocumentCleaner()
    return cleaner.run(documents=docs)["documents"]


def split_documents(docs: List[Document]) -> List[Document]:
    """Split by sentence for more coherent chunks, but skip already-structured docs."""
    # Separate docs that should NOT be split (curriculum, modulhandbuch have structured chunks)
    skip_split = []
    to_split = []
    
    for doc in docs:
        doctype = (doc.meta.get("doctype") or "").lower()
        chunk_type = doc.meta.get("chunk_type", "")
        # Skip splitting for structured documents
        if doctype in ["studienverlaufsplan", "modulhandbuch"] or chunk_type in ["module_overview", "module_detail"]:
            skip_split.append(doc)
        else:
            to_split.append(doc)
    
    if to_split:
        splitter = DocumentSplitter(
            split_by="sentence",
            split_length=5,        # 5 sentences per chunk
            split_overlap=1,       # 1 sentence overlap
        )
        splitter.warm_up()  # Required for sentence splitting (loads nltk)
        split_result = splitter.run(documents=to_split)["documents"]
    else:
        split_result = []
    
    print(f"  Skipped splitting for {len(skip_split)} structured docs")
    return skip_split + split_result


def deduplicate_chunks(docs: List[Document], similarity_threshold: float = 0.95) -> List[Document]:
    """Remove near-duplicate chunks based on content + metadata (program/degree)."""
    if not docs:
        return docs
    
    unique_docs = []
    seen_hashes = set()
    
    for doc in docs:
        meta = doc.meta or {}
        # Include program and degree in dedup key to avoid removing
        # same module content from different programs
        program = meta.get("program", "")
        degree = meta.get("degree", "")
        
        # Simple hash-based dedup on normalized content + metadata
        content_normalized = " ".join(doc.content.lower().split())
        # Use first 500 chars + program/degree for comparison
        content_key = f"{program}|{degree}|{content_normalized[:500]}"
        
        if content_key not in seen_hashes:
            seen_hashes.add(content_key)
            unique_docs.append(doc)
    
    removed = len(docs) - len(unique_docs)
    if removed > 0:
        print(f"Deduplizierung: {removed} Duplikate entfernt")
    
    return unique_docs


def embed_documents(docs: List[Document]) -> List[Document]:
    embedder = OpenAIDocumentEmbedder(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        api_base_url=OPENAI_EMBED_BASE_URL,  # Use local Ollama for embeddings
        model=OPENAI_EMBED_MODEL,
    )
    result = embedder.run(documents=docs)
    return result["documents"]


def write_documents(docs: List[Document]):
    store = get_document_store()
    writer = DocumentWriter(store, policy=DuplicatePolicy.OVERWRITE)
    writer.run(documents=docs)
    print(f"[INFO] Qdrant: {store.count_documents()} gespeicherte Chunks")



collector = MetadataCollector()
def build_embed_text(doc: Document) -> str:
    """
    Baut den finalen Text für Embeddings:
    - Injektion relevanter Metadaten als semantischer Kontext
    - bessere semantische Suche durch natürlichsprachige Beschreibung
    """
    meta = doc.meta or {}

    # Build natural language context for better embedding
    context_parts = []
    
    if meta.get("degree"):
        context_parts.append(f"Studienabschluss: {meta['degree']}")
    if meta.get("program"):
        context_parts.append(f"Studiengang: {meta['program']}")
    if meta.get("doctype"):
        context_parts.append(f"Dokumenttyp: {meta['doctype']}")
    if meta.get("module_name"):
        context_parts.append(f"Modul: {meta['module_name']}")
    if meta.get("module_id"):
        context_parts.append(f"Modul-ID: {meta['module_id']}")
    if meta.get("semester"):
        context_parts.append(f"Semester: {meta['semester']}")
    if meta.get("section"):
        context_parts.append(f"Abschnitt: {meta['section']}")
    if meta.get("paragraph"):
        context_parts.append(f"Paragraph: {meta['paragraph']}")
    if meta.get("ects"):
        context_parts.append(f"ECTS: {meta['ects']}")
    
    if not context_parts:
        return doc.content

    # Natural language header for better semantic matching
    context_header = "Kontext: " + ", ".join(context_parts) + "."
    return f"{context_header}\n\n{doc.content}"


def parse_pdf_with_correct_parser(path: Path, meta: Dict) -> List[Document]:
    """
    Wählt Parser, extrahiert Dokumente, updated Metadaten,
    injiziert für Retrieval relevante Metadaten in den Text,
    und sammelt alle Metas im MetadataCollector.
    """
    parser: BaseParser = select_parser(path.name)

    if parser is None:
        docs = fallback_extract(path, meta)
    else:
        print(f"[INFO] Parser gewaehlt: {parser.__class__.__name__} fuer {path.name}")
        docs = parser.parse(path, meta)

    for d in docs:
        d.meta = {**(d.meta or {}), **meta}

        d.content = normalize_text(d.content)

        collector.add(d.meta)

        d.content = build_embed_text(d)

    collector.add(meta)

    return docs



def index_pdfs_with_metadata():
    sources, metas = prepare_sources_and_meta()

    if not sources:
        print("[WARN] Keine Markdown-Dateien gefunden.")
        return

    # Filter to only include active (aktuell) documents
    filtered_sources = []
    filtered_metas = []
    for path, meta in zip(sources, metas):
        status = (meta.get("status") or "").lower()
        if status == "aktuell":
            filtered_sources.append(path)
            filtered_metas.append(meta)
        else:
            print(f"[SKIP] Archiviertes Dokument übersprungen: {path.name} (status={status})")

    print(f"[INFO] {len(filtered_sources)} aktuelle Dokumente von {len(sources)} gefunden")

    if not filtered_sources:
        print("[WARN] Keine aktiven Dokumente gefunden.")
        return

    print(f"[INFO] Starte parser-basierte Indexierung von {len(filtered_sources)} Markdown-Dateien...")

    all_docs: List[Document] = []

    for path, meta in zip(filtered_sources, filtered_metas):
        docs = parse_pdf_with_correct_parser(path, meta)
        all_docs.extend(docs)

    print(f"[INFO] Gesamte extrahierte Dokumente: {len(all_docs)}")

    cleaned = clean_documents(all_docs)

    # Split into sentence-based chunks for coherent retrieval
    splitted = split_documents(cleaned)
    print(f"[INFO] Nach Splitting: {len(splitted)} Chunks")

    # Remove near-duplicate chunks (e.g., similar PVO versions)
    deduped = deduplicate_chunks(splitted)
    print(f"[INFO] Nach Deduplizierung: {len(deduped)} Chunks")

    embedded = embed_documents(deduped)

    write_documents(embedded)

    summary = collector.summary()
    with open("metadata_cache.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[INFO] Metadata Cache gespeichert -> metadata_cache.json")
    print("[INFO] Indexierung abgeschlossen!")


if __name__ == "__main__":
    index_pdfs_with_metadata()
