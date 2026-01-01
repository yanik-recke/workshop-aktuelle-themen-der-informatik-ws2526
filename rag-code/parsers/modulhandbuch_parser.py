# parsers/modulhandbuch_parser.py
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict

import fitz  # PyMuPDF
from haystack.dataclasses import Document

from .base_parser import BaseParser

MAX_CHUNK_SIZE = 450  # embedding-optimiert, etwas größer erlaubt


class ModulhandbuchParser(BaseParser):
    """
    Parser für FH-Wedel Modulhandbücher.
    Extrahiert strukturierte Modulbereiche und erzeugt hochwertige RAG-Chunks.
    """

    doctypes = ["Modulhandbuch"]

    # Häufig vorkommende Abschnittsüberschriften
    SECTION_HEADERS = [
        r"Modulname", r"Modultitel", r"Modulnummer", r"Studiengang", r"Semester",
        r"Voraussetzungen", r"Lernziele", r"Inhalte?", r"Arbeitsaufwand",
        r"Prüfungsform", r"Literatur", r"Dozent(?:in)?", r"Lehrform"
    ]

    MODULE_START = re.compile(r"(Modulname|Modultitel)\s*[:\-]?", re.IGNORECASE)

    # ---------------------------------------------------------
    # PDF Extraction
    # ---------------------------------------------------------
    def _load_pdf_text(self, path: Path) -> str:
        """Extract raw PDF text."""
        doc = fitz.open(str(path))
        pages = [page.get_text("text") for page in doc]
        return "\n".join(pages)

    # ---------------------------------------------------------
    # Module Split
    # ---------------------------------------------------------
    def _split_into_modules(self, text: str) -> List[str]:
        """Split the entire document into module blocks."""
        blocks = []
        current = []

        for line in text.splitlines():
            if re.search(self.MODULE_START, line):
                if current:
                    blocks.append("\n".join(current))
                    current = []
            current.append(line)

        if current:
            blocks.append("\n".join(current))

        return blocks

    # ---------------------------------------------------------
    # Section Extraction
    # ---------------------------------------------------------
    def _extract_sections(self, module_text: str) -> Dict[str, str]:
        """
        Input: Textblock eines Moduls
        Output: { "Lernziele": "...", "Inhalte": "...", ... }
        """
        sections: Dict[str, List[str]] = {}
        current_title = "Einleitung"
        sections[current_title] = []

        for line in module_text.splitlines():
            stripped = line.strip()

            # Abschnitts-Header erkennen
            matched_header = None
            for header in self.SECTION_HEADERS:
                if re.match(header + r"\s*[:\-]?", stripped, flags=re.I):
                    matched_header = stripped
                    break

            if matched_header:
                # normalize header name
                current_title = re.sub(r"[:\-]\s*$", "", matched_header)
                sections[current_title] = []
            else:
                sections[current_title].append(stripped)

        # Join blocks
        return {title: "\n".join(v).strip() for title, v in sections.items()}

    # ---------------------------------------------------------
    # Chunking
    # ---------------------------------------------------------
    def _section_to_chunks(
        self, 
        module_name: str, 
        sections: Dict[str, str],
        program: str = "",
        degree: str = "",
    ) -> List[Document]:
        chunks: List[Document] = []
        
        # Context header to include in each chunk for better retrieval
        context_header = f"Modul: {module_name}\nStudiengang: {program} ({degree})\n\n"

        for subsection, content in sections.items():
            if not content.strip():
                continue

            lines = content.split("\n")
            current = []
            length = len(context_header)  # Account for header

            for line in lines:
                if length + len(line) > MAX_CHUNK_SIZE:
                    chunk_content = context_header + f"Abschnitt: {subsection}\n" + "\n".join(current)
                    chunks.append(
                        Document(
                            content=chunk_content,
                            meta={
                                "module_name": module_name,
                                "section": subsection,
                            }
                        )
                    )
                    current = []
                    length = len(context_header)

                current.append(line)
                length += len(line)

            if current:
                chunk_content = context_header + f"Abschnitt: {subsection}\n" + "\n".join(current)
                chunks.append(
                    Document(
                        content=chunk_content,
                        meta={
                            "module_name": module_name,
                            "section": subsection,
                        }
                    )
                )

        return chunks

    # ---------------------------------------------------------
    # Main parse()
    # ---------------------------------------------------------
    def parse(self, path: Path, meta: Dict) -> List[Document]:
        """
        BaseParser-konform:
        - nimmt path als Path
        - gibt List[Document] zurück
        - erzeugt strukturierte Chunks
        - setzt Metadaten
        """
        text = self._load_pdf_text(path)
        module_blocks = self._split_into_modules(text)

        documents: List[Document] = []

        base_meta = self._base_meta(meta)
        program = base_meta.get("program", "Unbekannter Studiengang")
        degree = base_meta.get("degree", "Unbekannter Abschluss")
        
        # Collect all module names for summary
        all_modules = []

        for block in module_blocks:
            sections = self._extract_sections(block)
            module_name = (
                sections.get("Modulname")
                or sections.get("Modultitel")
                or "Unbekanntes Modul"
            ).strip()
            
            all_modules.append(module_name)

            section_docs = self._section_to_chunks(
                module_name, 
                sections,
                program=program,
                degree=degree,
            )

            # Metadaten anreichern
            for d in section_docs:
                d.meta = {**base_meta, **d.meta}

            documents.extend(section_docs)
        
        # Create a summary document listing all modules in this handbook
        if all_modules:
            summary_content = f"""Modulhandbuch für {program} ({degree}) an der FH Wedel
Studiengang: {program}
Abschluss: {degree}
Anzahl Module: {len(all_modules)}

Liste aller Module im Modulhandbuch:
""" + "\n".join(f"  - {m}" for m in all_modules if m != "Unbekanntes Modul")
            
            summary_meta = dict(base_meta)
            summary_meta["chunk_type"] = "module_overview"
            documents.insert(0, Document(content=summary_content, meta=summary_meta))

        return documents
