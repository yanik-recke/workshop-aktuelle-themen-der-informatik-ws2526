# parsers/curriculum_parser.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import pdfplumber
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


class CurriculumParser(BaseParser):
    doctypes = ["Studienverlaufsplan"]

    def parse(self, path: Path, meta: Dict) -> List[Document]:
        base_meta = self._base_meta(meta)

        modules: List[Dict] = []

        current_semester: Optional[int] = None
        current_module: Optional[Dict] = None

        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
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

        for mod in modules:
            content_lines = [
                f"Studienverlaufsplan FH Wedel",
                f"Studiengang: {program} ({degree})",
            ]

            if mod.get("semester"):
                content_lines.append(f"Empfohlenes Semester: {mod['semester']}. Semester")

            content_lines.append(f"Modul {mod['code']}: {mod['title']}")
            content_lines.append("")
            content_lines.append("Tabellenzeilen:")
            content_lines.extend(mod["lines"])

            content = "\n".join(content_lines).strip()

            doc_meta = dict(base_meta)
            doc_meta.update({
                "module_code": mod["code"],
                "module_title": mod["title"],
                "semester": mod.get("semester"),
            })

            documents.append(Document(content=content, meta=doc_meta))

        return documents
