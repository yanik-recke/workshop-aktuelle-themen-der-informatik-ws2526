# parsers/curriculum_parser.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

from haystack.dataclasses import Document

from .base_parser import BaseParser

# Old Bachelor-style blocks: MBxxx
MB_RX = re.compile(r"^(MB\d{3})\s+(.+)$")
TB_RX = re.compile(r"^(TB\d{3})\s+(.+)$")

# New Master-style blocks: MMxxx
MM_RX = re.compile(r"^(MM\d{3})\s+(.+)$")
TM_RX = re.compile(r"^(TM\d{3})\s+(.+)$")

# Semester column detection (Master curricula)
SEMESTER_COL_RX = re.compile(r"[WS]\s+(\d)\s+\d{1,3}")

# Explicit semester headers (Bachelor curricula)
SEMESTER_HEADER_RX = re.compile(r"^\s*(\d)\.\s*Semester\b", re.IGNORECASE)

# Legend for curriculum table columns (helps LLM understand abbreviations)
CURRICULUM_LEGEND = """
Legende der Spalten:
- ECTS: Credit Points
- Fq: Frequenz (W=Wintersemester, S=Sommersemester)
- VE: Veranstaltungseinheit (75 Min/Woche)
- KoZ: Kontaktzeit, EiZ: Selbststudium, AA: Arbeitsaufwand
- Anw: Anwesenheit, Vorl: Vorleistungen
- Art: Prüfungsform (KL=Klausur, MP=Mündl., PFK=Portfolio)
- LF: Veranstaltungsform (V=Vorlesung, VU=Vorlesung+Übung, P=Praktikum, S=Seminar)
""".strip()


class CurriculumParser(BaseParser):
    doctypes = ["Studienverlaufsplan"]

    def parse(self, path: Path, meta: Dict) -> List[Document]:
        base_meta = self._base_meta(meta)

        modules: List[Dict] = []

        current_semester: Optional[int] = None
        current_module: Optional[Dict] = None

        # Read markdown file instead of PDF
        text = self._read_markdown(path)
        lines = text.splitlines()

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            # -------------------------------
            # 1) Bachelor-style "1. Semester"
            # -------------------------------
            m_sem_header = SEMESTER_HEADER_RX.match(line)
            if m_sem_header:
                current_semester = int(m_sem_header.group(1))
                continue

            # -------------------------------
            # 2) Bachelor-style MB-Module
            # -------------------------------
            m_mb = MB_RX.match(line)
            if m_mb:
                if current_module:
                    modules.append(current_module)

                current_module = {
                    "code": m_mb.group(1),
                    "title": m_mb.group(2),
                    "semester": current_semester,
                    "lines": [],
                }
                continue

            # bachelor TB rows
            m_tb = TB_RX.match(line)
            if m_tb and current_module:
                current_module["lines"].append(line)
                continue

            # -------------------------------
            # 3) Master-style MM module block
            # -------------------------------
            m_mm = MM_RX.match(line)
            if m_mm:
                if current_module:
                    modules.append(current_module)

                current_module = {
                    "code": m_mm.group(1),
                    "title": m_mm.group(2),
                    "semester": None,
                    "lines": [],
                }
                continue

            # -------------------------------
            # 4) Master-style TM (Teilmodule)
            # -------------------------------
            m_tm = TM_RX.match(line)
            if m_tm and current_module:
                # Try to extract semester from table columns
                m_semcol = SEMESTER_COL_RX.search(line)
                semester = int(m_semcol.group(1)) if m_semcol else None

                current_module["lines"].append(line)

                if semester and not current_module.get("semester"):
                    current_module["semester"] = semester

                continue

            # -------------------------------
            # 5) Other lines inside module
            # -------------------------------
            if current_module:
                current_module["lines"].append(line)

        # final block
        if current_module:
            modules.append(current_module)

        # --------------------------
        # Build Haystack documents
        # --------------------------
        documents = []
        program = base_meta.get("program", "Unbekannter Studiengang")
        degree = base_meta.get("degree", "Unbekannter Abschluss")

        # 1) Create a SUMMARY document listing ALL modules for this program
        #    This helps with "which modules are in program X?" queries
        if modules:
            summary_lines = [
                f"Modulübersicht für {program} ({degree}) an der FH Wedel",
                f"Studiengang: {program}",
                f"Abschluss: {degree}",
                f"Anzahl Module: {len(modules)}",
                "",
                CURRICULUM_LEGEND,
                "",
                "Liste aller Module:",
            ]
            
            # Group by semester
            by_semester = {}
            for mod in modules:
                sem = mod.get("semester") or 0
                if sem not in by_semester:
                    by_semester[sem] = []
                by_semester[sem].append(mod)
            
            for sem in sorted(by_semester.keys()):
                if sem > 0:
                    summary_lines.append(f"\n{sem}. Semester:")
                else:
                    summary_lines.append(f"\nWahlmodule/Sonstige:")
                for mod in by_semester[sem]:
                    summary_lines.append(f"  - {mod['code']}: {mod['title']}")
            
            summary_content = "\n".join(summary_lines)
            summary_meta = dict(base_meta)
            summary_meta["chunk_type"] = "module_overview"
            documents.append(Document(content=summary_content, meta=summary_meta))

        # 2) Create individual module documents with full context
        for mod in modules:
            content_lines = [
                f"Modul im Studiengang {program} ({degree}) an der FH Wedel",
                f"Studiengang: {program}",
                f"Abschluss: {degree}",
                "",
                CURRICULUM_LEGEND,
            ]

            if mod.get("semester"):
                content_lines.append(f"Empfohlenes Semester: {mod['semester']}. Semester")

            content_lines.append(f"\nModul {mod['code']}: {mod['title']}")
            
            if mod["lines"]:
                content_lines.append("")
                content_lines.append("Details:")
                content_lines.extend(mod["lines"])

            content = "\n".join(content_lines).strip()

            doc_meta = dict(base_meta)
            doc_meta.update({
                "module_code": mod["code"],
                "module_title": mod["title"],
                "semester": mod.get("semester"),
                "chunk_type": "module_detail",
            })

            documents.append(Document(content=content, meta=doc_meta))

        return documents
