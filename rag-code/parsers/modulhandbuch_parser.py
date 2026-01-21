# parsers/modulhandbuch_parser.py
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Tuple

from haystack.dataclasses import Document

from .base_parser import BaseParser

MAX_CHUNK_SIZE = 2000  # Characters per chunk


class ModulhandbuchParser(BaseParser):
    """
    Parser für FH-Wedel Modulhandbücher.
    Parses markdown files with format like:
    - **MB001 – Analysis**
    |Verantwortliche:|Name|
    ...
    """

    doctypes = ["Modulhandbuch"]

    # Pattern to match module headers like "- **MB001 – Analysis**" or "**MB001 – Analysis**"
    MODULE_HEADER_RX = re.compile(r"^-?\s*\*\*([MT]B\d{3})\s*[–-]\s*(.+?)\*\*\s*$")

    # Pattern to match table of contents entries like "MB001 – Analysis"
    TOC_ENTRY_RX = re.compile(r"^([MT]B\d{3})\s*[–-]\s*(.+)$")

    def _load_text(self, path: Path) -> str:
        """Read markdown file text."""
        return self._read_markdown(path)

    def _extract_modules(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract modules from the markdown text.
        Returns list of (module_code, module_name, module_content).
        """
        lines = text.splitlines()
        modules = []
        current_code = None
        current_name = None
        current_content = []
        in_toc = True  # Start assuming we're in table of contents

        for line in lines:
            # Check if this is a module header (marks end of TOC)
            header_match = self.MODULE_HEADER_RX.match(line.strip())
            if header_match:
                in_toc = False
                # Save previous module if exists
                if current_code and current_content:
                    modules.append((current_code, current_name, "\n".join(current_content)))

                current_code = header_match.group(1)
                current_name = header_match.group(2).strip()
                current_content = [line]
                continue

            # Skip TOC entries
            if in_toc:
                continue

            # Add line to current module content
            if current_code:
                current_content.append(line)

        # Don't forget the last module
        if current_code and current_content:
            modules.append((current_code, current_name, "\n".join(current_content)))

        return modules

    def _extract_toc_modules(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract module list from table of contents.
        Returns list of (module_code, module_name).
        """
        modules = []
        lines = text.splitlines()

        for line in lines:
            # Stop at "# **Module**" section
            if line.strip().startswith("# **Module"):
                break

            match = self.TOC_ENTRY_RX.match(line.strip())
            if match:
                modules.append((match.group(1), match.group(2).strip()))

        return modules

    def parse(self, path: Path, meta: Dict) -> List[Document]:
        """
        Parse Modulhandbuch markdown file.
        Creates:
        1. A summary document with all module names
        2. Individual documents for each module
        """
        text = self._load_text(path)

        base_meta = self._base_meta(meta)
        program = base_meta.get("program", "Unbekannter Studiengang")
        degree = base_meta.get("degree", "Unbekannter Abschluss")

        documents: List[Document] = []

        # Extract modules from TOC for the summary
        toc_modules = self._extract_toc_modules(text)

        # Extract full module content
        full_modules = self._extract_modules(text)

        # Use TOC for module names if available, otherwise use extracted modules
        module_names = [f"{code} – {name}" for code, name in toc_modules] if toc_modules else [f"{code} – {name}" for code, name, content in full_modules]

        # Create summary document
        if module_names:
            summary_content = f"""Modulhandbuch für {program} ({degree}) an der FH Wedel
Studiengang: {program}
Abschluss: {degree}
Anzahl Module: {len(module_names)}

Liste aller Module im Modulhandbuch:
""" + "\n".join(f"  - {name}" for name in module_names)

            summary_meta = dict(base_meta)
            summary_meta["chunk_type"] = "module_overview"
            documents.append(Document(content=summary_content, meta=summary_meta))

        # Create individual module documents
        for code, name, content in full_modules:
            # Clean up content - remove excessive whitespace
            cleaned_content = re.sub(r'\n{3,}', '\n\n', content)

            # Truncate if needed
            if len(cleaned_content) > MAX_CHUNK_SIZE:
                cleaned_content = cleaned_content[:MAX_CHUNK_SIZE] + "..."

            doc_content = f"""Modul: {code} – {name}
Studiengang: {program} ({degree})

{cleaned_content}"""

            doc_meta = dict(base_meta)
            doc_meta["module_code"] = code
            doc_meta["module_name"] = name
            doc_meta["chunk_type"] = "module_detail"

            documents.append(Document(content=doc_content, meta=doc_meta))

        return documents
