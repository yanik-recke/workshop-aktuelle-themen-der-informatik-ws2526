from __future__ import annotations
from pathlib import Path
import json
import re
from typing import Dict, List, Tuple

from haystack.dataclasses import Document

from config import DATA_DIR, META_FILE


def load_metadata(meta_file: Path) -> Dict[str, dict]:
    """
    Lädt fhwedel_docs.json und baut ein Lookup:
      filename -> metadata-dict
    """
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_file}")

    with open(meta_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    lookup: Dict[str, dict] = {}
    for entry in raw:
        filename = Path(entry["local_path"]).name
        lookup[filename] = entry

    print(f"🔎 Metadaten geladen: {len(lookup)} Einträge")
    return lookup



def find_pdfs_with_metadata(base_dir: Path, metadb: Dict[str, dict]) -> List[Dict]:
    """
    Sucht alle PDFs im DATA_DIR und ordnet ihnen die Metadaten aus fhwedel_docs.json zu.
    Liefert Liste:
      [{"path": Path, "meta": {...}}, ...]
    """
    pdfs = list(base_dir.rglob("*.pdf"))
    results: List[Dict] = []

    for path in pdfs:
        filename = path.name
        meta = metadb.get(filename)
        if not meta:
            print(f"⚠️ Warnung: Keine Metadaten zu {filename} gefunden")
            continue

        if meta.get("status") and meta["status"] != "aktuell" or "Moduluebersicht".upper() in meta.get("filename").upper():
            continue

        results.append({
            "path": path,
            "meta": {
                "filename": meta.get("filename"),
                "degree": meta.get("degree"),
                "program": meta.get("program"),
                "doctype": meta.get("doctype"),
                "status": meta.get("status"),
                "version": meta.get("version"),
                "url": meta.get("url"),
            }
        })

    print(f"📚 PDFs mit Metadaten: {len(results)}")
    return results



_ABBREVIATIONS: List[Tuple[str, str]] = [
    (r"\bECTS\b", "ECTS (European Credit Transfer System)"),
    (r"\bSWS\b", "Semesterwochenstunden (SWS)"),
    (r"\bK1\b", "Klausur (K1)"),
    (r"\bK2\b", "Klausur (K2)"),
    (r"\bPF\b", "Portfolio-Prüfung (PF)"),
    (r"\bPL\b", "Prüfungsleistung (PL)"),
]


def normalize_text(text: str) -> str:
    """
    Ersetzt typische Abkürzungen durch leicht erklärende Varianten.
    Das verbessert die semantische Dichte der Embeddings.
    """
    norm = text
    for pattern, repl in _ABBREVIATIONS:
        norm = re.sub(pattern, repl, norm)
    return norm



def enrich_document_for_embedding(doc: Document) -> Document:
    """
    Baut aus Metadaten + Originaltext einen embedding-freundlichen Content:

    [Degree: BWL] [Program: Betriebswirtschaftslehre] [DocType: Modulhandbuch] [Status: aktuell] [Version: B_BWL23.0]
    [URL: https://...]

    <bereinigter Text>

    Wichtig:
    - keine neuen Fakten erfinden
    - nur Metadaten aus doc.meta verwenden
    """
    meta = doc.meta or {}

    header_parts = []

    degree = meta.get("degree")
    if degree:
        header_parts.append(f"Degree: {degree}")

    program = meta.get("program")
    if program:
        header_parts.append(f"Program: {program}")

    doctype = meta.get("doctype")
    if doctype:
        header_parts.append(f"DocType: {doctype}")

    status = meta.get("status")
    if status:
        header_parts.append(f"Status: {status}")

    version = meta.get("version")
    if version:
        header_parts.append(f"Version: {version}")

    url = meta.get("url")
    if url:
        header_parts.append(f"URL: {url}")

    header_line = " | ".join(header_parts) if header_parts else ""

    cleaned = normalize_text(doc.content or "")

    if header_line:
        new_content = header_line + "\n\n" + cleaned
    else:
        new_content = cleaned

    doc.content = new_content
    return doc


def enrich_documents_for_embedding(docs: List[Document]) -> List[Document]:
    """
    Wendet enrich_document_for_embedding auf alle Dokumente an.
    """
    return [enrich_document_for_embedding(d) for d in docs]



def prepare_sources_and_meta() -> Tuple[List[Path], List[dict]]:
    """
    Wird von der Indexing-Pipeline verwendet:
    - lädt Metadaten
    - findet passende PDFs
    - gibt zwei parallele Listen zurück:
        sources: List[Path]
        metas:   List[dict]
    """
    metadb = load_metadata(Path(META_FILE))
    entries = find_pdfs_with_metadata(DATA_DIR, metadb)
    # entries = entries[:10]  # Removed limit - index all PDFs
    sources = [e["path"] for e in entries]
    metas = [e["meta"] for e in entries]
    return sources, metas
