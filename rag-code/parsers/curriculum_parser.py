# parsers/curriculum_parser.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from haystack.dataclasses import Document

from .base_parser import BaseParser

# Module code patterns (in table cells)
MODULE_CODE_RX = re.compile(r'(M[BM]\d{3})')
SUBMODULE_CODE_RX = re.compile(r'(T[BM]\d{3})')



class CurriculumParser(BaseParser):
    doctypes = ["Studienverlaufsplan"]

    def _parse_table_row(self, line: str) -> List[str]:
        """Parse a markdown table row into cells."""
        if not line.startswith('|'):
            return []
        # Split by | and clean up
        cells = [c.strip() for c in line.split('|')]
        # Remove empty first/last from leading/trailing |
        if cells and cells[0] == '':
            cells = cells[1:]
        if cells and cells[-1] == '':
            cells = cells[:-1]
        return cells

    def _extract_module_info(self, cell: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract module code and title from a cell with <br> separated lines.
        
        Handles formats like:
        - "MB001 Analysis<br>TB001<br>..." -> code=MB001, title=Analysis
        - "MB359<br>TB353<br>Study Bootcamp Data Science<br>..." -> code=MB359, title=Study Bootcamp Data Science
        """
        if not cell:
            return None, None
        
        lines = [l.strip() for l in cell.split('<br>')]
        code = None
        title = None
        
        for line in lines:
            # Look for module code (MB123 or MM123)
            code_match = MODULE_CODE_RX.search(line)
            if code_match and not code:
                code = code_match.group(1)
                # Title might be on same line after the code
                rest = line[code_match.end():].strip()
                if rest and not SUBMODULE_CODE_RX.match(rest):
                    title = rest
            # If we have a code but no title yet, look for title in subsequent lines
            elif code and not title and line:
                # Skip submodule codes (TB123)
                if SUBMODULE_CODE_RX.match(line):
                    continue
                # Skip lines that look like headers or formatting
                if line.startswith('**') or line.startswith('Prfg'):
                    continue
                # This is likely the title
                if len(line) > 3:
                    title = line
                    break
        
        return code, title

    def _detect_semester(self, cells: List[str]) -> Optional[int]:
        """Detect which semester a module belongs to based on ECTS columns."""
        # Based on table structure: cells[3] = Sem 1, cells[4] = Sem 2, ..., cells[9] = Sem 7
        if len(cells) < 10:
            return None
        for semester, idx in enumerate(range(3, 10), start=1):
            if idx >= len(cells):
                break
            cell_clean = cells[idx].replace(',', '.').strip()
            try:
                val = float(cell_clean)
                if val > 0:
                    return semester
            except (ValueError, IndexError):
                continue
        return None

    def _extract_ects(self, cells: List[str]) -> Optional[float]:
        """Extract ECTS value from a row (sum of semester columns 3-9)."""
        if len(cells) < 10:
            return None
        total = 0.0
        for idx in range(3, 10):
            if idx >= len(cells):
                break
            cell_clean = cells[idx].replace(',', '.').strip()
            try:
                val = float(cell_clean)
                if val > 0:
                    total += val
            except (ValueError, IndexError):
                continue
        return total if total > 0 else None

    def parse(self, path: Path, meta: Dict) -> List[Document]:
        base_meta = self._base_meta(meta)
        modules: List[Dict] = []
        module_semesters: Dict[str, int] = {}  # Track semester per module code
        module_ects: Dict[str, float] = {}  # Track ECTS per module code

        # Read markdown file
        text = self._read_markdown(path)
        lines = text.splitlines()

        # First pass: collect all modules, semesters, and ECTS from detail rows
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith('|---'):
                continue
            
            cells = self._parse_table_row(line)
            if len(cells) < 11:
                continue
            
            # Check for module code in any cell
            for cell in cells[:3]:
                module_match = MODULE_CODE_RX.search(cell)
                if module_match:
                    code = module_match.group(1)
                    # Try to detect semester from this row
                    semester = self._detect_semester(cells)
                    if semester and code not in module_semesters:
                        module_semesters[code] = semester
                    # Accumulate ECTS from detail rows
                    ects = self._extract_ects(cells)
                    if ects:
                        module_ects[code] = module_ects.get(code, 0) + ects
                    break

        # Second pass: build module list
        seen_codes = set()
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith('|---'):
                continue
            
            cells = self._parse_table_row(line)
            if len(cells) < 4:
                continue
            
            first_cell = cells[0] if cells else ""
            code, title = self._extract_module_info(first_cell)
            
            if code and code not in seen_codes:
                seen_codes.add(code)
                
                if not title:
                    for c in cells[1:3]:
                        if c and not c.startswith('**') and len(c) > 3:
                            clean = c.replace('<br>', ' ').strip()
                            if clean and not MODULE_CODE_RX.match(clean):
                                title = clean
                                break
                
                if not title:
                    title = code
                
                # Get semester and ECTS from our collected data
                semester = module_semesters.get(code)
                ects = module_ects.get(code)
                
                modules.append({
                    "code": code,
                    "title": title,
                    "semester": semester,
                    "ects": ects,
                    "lines": [],
                })

        # --------------------------
        # Build Haystack documents
        # --------------------------
        documents = []
        program = base_meta.get("program", "Unbekannter Studiengang")
        degree = base_meta.get("degree", "Unbekannter Abschluss")

        # Group modules by semester first
        by_semester = {}
        for mod in modules:
            sem = mod.get("semester") or 0
            if sem not in by_semester:
                by_semester[sem] = []
            by_semester[sem].append(mod)
        
        # 1) Create a SUMMARY document listing ALL modules for this program
        if modules:
            summary_lines = [
                f"Modulübersicht für {program} ({degree}) an der FH Wedel",
                f"Studiengang: {program}",
                f"Abschluss: {degree}",
                f"Anzahl Module: {len(modules)}",
                "",
                "Liste aller Module:",
            ]
            
            for sem in sorted(by_semester.keys()):
                if sem > 0:
                    summary_lines.append(f"\n{sem}. Semester:")
                else:
                    summary_lines.append(f"\nWahlmodule/Sonstige:")
                for mod in by_semester[sem]:
                    ects_str = f" ({mod['ects']:.0f} ECTS)" if mod.get('ects') else ""
                    summary_lines.append(f"  - {mod['code']}: {mod['title']}{ects_str}")
            
            summary_content = "\n".join(summary_lines)
            summary_meta = dict(base_meta)
            summary_meta["chunk_type"] = "module_overview"
            documents.append(Document(content=summary_content, meta=summary_meta))
        
        # 2) Create PER-SEMESTER summary documents for semester-specific queries
        for sem, sem_modules in by_semester.items():
            if sem == 0:
                continue  # Skip optional/elective modules
            
            sem_lines = [
                f"Module im {sem}. Semester - {program} ({degree}) an der FH Wedel",
                f"Studiengang: {program}",
                f"Abschluss: {degree}",
                f"Semester: {sem}",
                f"Anzahl Module in diesem Semester: {len(sem_modules)}",
                "",
                f"Module im {sem}. Semester:",
            ]
            
            sem_ects_total = sum(m.get('ects', 0) or 0 for m in sem_modules)
            if sem_ects_total > 0:
                sem_lines.append(f"Gesamt ECTS im {sem}. Semester: {sem_ects_total:.0f}")
            sem_lines.append("")
            
            for mod in sem_modules:
                ects_str = f" ({mod['ects']:.0f} ECTS)" if mod.get('ects') else ""
                sem_lines.append(f"- {mod['code']}: {mod['title']}{ects_str}")
            
            sem_content = "\n".join(sem_lines)
            sem_meta = dict(base_meta)
            sem_meta["chunk_type"] = "semester_overview"
            sem_meta["semester"] = sem
            documents.append(Document(content=sem_content, meta=sem_meta))

        # 3) Create individual module documents with full context
        for mod in modules:
            content_lines = [
                f"Modul im Studiengang {program} ({degree}) an der FH Wedel",
                f"Studiengang: {program}",
                f"Abschluss: {degree}",
            ]

            if mod.get("semester"):
                content_lines.append(f"Empfohlenes Semester: {mod['semester']}. Semester")
            
            if mod.get("ects"):
                content_lines.append(f"ECTS: {mod['ects']:.0f}")

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
