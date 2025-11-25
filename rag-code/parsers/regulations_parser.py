
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import pdfplumber
from haystack.dataclasses import Document

from .base_parser import BaseParser


class RegulationsParser(BaseParser):

    doctypes = ["ZLO", "PVO", "SPO"]  


    RE_SECTION = re.compile(r"^\s*([IVXLC]+\.)\s+(.*)")
    RE_PARAGRAPH = re.compile(r"^\s*§\s*(\d+)\s*(.*)")
    RE_ABS = re.compile(r"^\s*\((\d+)\)\s+(.*)")

    def parse(self, path: Path, meta: Dict) -> List[Document]:
        base_meta = self._base_meta(meta)
        docs: List[Document] = []

        text = self._extract_text(path)

        # Split into lines to evaluate paragraph structure
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        current_section: Optional[str] = None
        current_paragraph_num: Optional[str] = None
        current_paragraph_title: Optional[str] = None
        buffer: List[str] = []

        def flush_paragraph():
            """Speichert den aktuellen Paragraph als Document."""
            if not buffer or not current_paragraph_num:
                return

            content = "\n".join(buffer).strip()
            if not content:
                return

            docs.append(
                Document(
                    content=content,
                    meta={
                        **base_meta,
                        "section": current_section,
                        "paragraph": current_paragraph_num,
                        "paragraph_title": current_paragraph_title,
                    },
                )
            )

        for line in lines:

            sec = self.RE_SECTION.match(line)
            if sec:
                flush_paragraph()
                current_section = sec.group(2).strip()
                current_paragraph_num = None
                current_paragraph_title = None
                buffer = []
                continue
            para = self.RE_PARAGRAPH.match(line)

            if para:
                flush_paragraph()
                current_paragraph_num = f"§{para.group(1)}"
                current_paragraph_title = para.group(2).strip() or None
                buffer = [line]  # Keep paragraph header
                continue
            absatz = self.RE_ABS.match(line)
            if absatz and current_paragraph_num:
                buffer.append(line)
                continue

            if current_paragraph_num:
                buffer.append(line)

        # final flush
        flush_paragraph()

        return docs

    def _extract_text(self, path: Path) -> str:
        """Extract text from PDF using pdfplumber with cleanup."""
        all_text = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                all_text.append(txt)

        text = "\n".join(all_text)

        # Clean up repeated page headers/footers
        text = re.sub(r"Seite \d+ von \d+", "", text)
        text = re.sub(r"Fachhochschule Wedel.*", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
