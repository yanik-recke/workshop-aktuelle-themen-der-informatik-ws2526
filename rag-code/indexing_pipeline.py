from __future__ import annotations

import json
import ntpath
from pathlib import Path
from typing import List, Dict

from haystack.dataclasses import Document
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret

from config import OPENAI_EMBED_MODEL
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

from haystack.components.converters.pypdf import PyPDFToDocument
pypdf_converter = PyPDFToDocument()


def fallback_extract(path: Path, meta: Dict) -> List[Document]:
    result = pypdf_converter.run(sources=[path], meta=[meta])
    docs = result["documents"]
    print(f"⚠️ Fallback-Parser benutzt für {path.name} ({len(docs)} Seiten)")
    return docs



def clean_documents(docs: List[Document]) -> List[Document]:
    cleaner = DocumentCleaner()
    return cleaner.run(documents=docs)["documents"]


def embed_documents(docs: List[Document]) -> List[Document]:
    embedder = OpenAIDocumentEmbedder(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        model=OPENAI_EMBED_MODEL,
    )
    result = embedder.run(documents=docs)
    return result["documents"]


def write_documents(docs: List[Document]):
    store = get_document_store()
    writer = DocumentWriter(store, policy=DuplicatePolicy.OVERWRITE)
    writer.run(documents=docs)
    print(f"💾 Qdrant: {store.count_documents()} gespeicherte Chunks")



collector = MetadataCollector()
def build_embed_text(doc: Document) -> str:
    """
    Baut den finalen Text für Embeddings:
    - Injektion relevanter Metadaten als Präfix
    - bessere semantische Suche (Semester, Modul-ID, Degree, Program, etc.)
    """
    meta = doc.meta or {}

    relevant_keys = [
        "degree", "program", "doctype", "module_id", "module_name",
        "ects", "semester", "track", "status", "version"
    ]

    meta_text_parts = []
    for k in relevant_keys:
        v = meta.get(k)
        if isinstance(v, (str, int, float)) and str(v).strip():
            meta_text_parts.append(f"{k.capitalize()}: {v}")

    meta_prefix = " | ".join(meta_text_parts)

    if not meta_prefix:
        return doc.content

    return f"{meta_prefix}\n\n{doc.content}"


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
        print(f"🧩 Parser gewählt: {parser.__class__.__name__} für {path.name}")
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
        print("⚠️ Keine PDFs gefunden.")
        return

    print(f"🚀 Starte parser-basierte Indexierung von {len(sources)} PDFs...")

    all_docs: List[Document] = []

    for path, meta in zip(sources, metas):
        docs = parse_pdf_with_correct_parser(path, meta)
        all_docs.extend(docs)

    print(f"📄 Gesamte extrahierte Dokumente: {len(all_docs)}")

    cleaned = clean_documents(all_docs)

    embedded = embed_documents(cleaned)

    write_documents(embedded)

    summary = collector.summary()
    with open("metadata_cache.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("📦 Metadata Cache gespeichert → metadata_cache.json")
    print("🎉 Indexierung abgeschlossen!")


if __name__ == "__main__":
    index_pdfs_with_metadata()
