# metadata_collector.py
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Set, List

class MetadataCollector:
    """
    Sammelt Metadaten aus allen Parsern und stellt sie später der
    Query Expansion und Context Detection zur Verfügung.
    """

    def __init__(self):
        self.data: Dict[str, Set[str]] = defaultdict(set)

    def add(self, meta: Dict):
        """
        Füge neue Metadaten aus einem Dokument hinzu.
        """
        for key, value in meta.items():
            if not value:
                continue
            if isinstance(value, str):
                self.data[key].add(value.strip())
            elif isinstance(value, list):
                for v in value:
                    self.data[key].add(str(v).strip())

    def add_bulk(self, metas: List[Dict]):
        for m in metas:
            self.add(m)

    def summary(self) -> Dict[str, List[str]]:
        """
        Exportiere alles in Listenform.
        Ideal für Query Expansion & Context Detection.
        """
        return {key: sorted(values) for key, values in self.data.items()}

    def __repr__(self):
        return f"MetadataCollector(keys={list(self.data.keys())})"
