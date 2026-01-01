# parsers/base_parser.py
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

from haystack.dataclasses import Document


class BaseParser(ABC):
    """
    Abstrakte Basis-Klasse für alle FH-Wedel-spezifischen PDF-Parser.

    Idee:
    - can_handle(meta): entscheidet, ob dieser Parser für das Dokument zuständig ist
      (z.B. anhand von meta["doctype"])
    - parse(path, meta): erzeugt eine Liste von Haystack-Documents mit sinnvollem,
      für RAG geeigneten Text.
    """

    # Liste der unterstützten Doctypes (z.B. ["Studienverlaufsplan"])
    doctypes: List[str] = []

    def can_handle(self, meta: Dict) -> bool:
        """
        Default-Implementierung: prüfe, ob der Doctype in self.doctypes ist.
        """
        doctype = (meta.get("doctype") or "").strip()
        return bool(doctype) and doctype in self.doctypes

    @abstractmethod
    def parse(self, path: Path, meta: Dict) -> List[Document]:
        """
        Liest die PDF-Datei und gibt eine Liste von Haystack-Documents zurück.
        """
        raise NotImplementedError("Subclasses must implement parse()")

    # Kleine Hilfsfunktion für ein konsistentes Meta-Mapping
    def _base_meta(self, meta: Dict) -> Dict:
        return {
            "degree": meta.get("degree"),
            "program": meta.get("program"),
            "doctype": meta.get("doctype"),
            "status": meta.get("status"),
            "version": meta.get("version"),
            "url": meta.get("url"),
        }
