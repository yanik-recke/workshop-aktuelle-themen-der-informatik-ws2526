import os
import re
import csv
import json
import unicodedata
import asyncio
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm

import ssl  
import certifi 
import aiohttp
import aiofiles  
import pymupdf4llm  
# from marker.converters.pdf import PdfConverter
# from marker.models import create_model_dict
# from marker.output import text_from_rendered
# ========================================
# Konfiguration
# ========================================

BASE_URL = "https://www.fh-wedel.de"

PRUEFUNGSORDNUNGEN_URL = (
    "https://www.fh-wedel.de/studieren/pruefungscenter/pruefungsordnungen/"
)

# Studiengangs-spezifische Seiten (Beispiel – hier einfach Platzhalter)
PROGRAM_PAGE_URLS: List[str] = [
    "https://www.fh-wedel.de/bewerben/bachelor/informatik/",
    "https://www.fh-wedel.de/bewerben/bachelor/medieninformatik/",
]

# neue Verzeichnisstruktur
DATA_BASE_DIR = os.path.join("data", "documents")
ORIGINAL_DIR = os.path.join(DATA_BASE_DIR, "original-docs")
MD_DIR = os.path.join(DATA_BASE_DIR, "md-docs")
META_JSON = os.path.join(DATA_BASE_DIR, "meta.json")
META_CSV = os.path.join(DATA_BASE_DIR, "meta.csv")

# ========================================
# Musterdefinitionen
# ========================================
PROGCODE_RX = re.compile(
    r'(?<![A-Za-z0-9])([BM])[_\- ]([A-ZÄÖÜa-zäöü]{2,12})(?=\d|_|\.|$)',
    re.UNICODE
)
# in Dateinamen / Link-Texten: z.B. 23.0, 20.1a, 1.0, …
VERSION_RX = re.compile(r'(\d{2})\.(\d{1,2})([a-zA-Z]?)')

DOCTYPE_PATTERNS = [
    (re.compile(r'\bSPO\b', re.IGNORECASE), "Studien- und Pruefungsordnung"),
    (re.compile(r'\bModulhandbuch\b', re.IGNORECASE), "Modulhandbuch"),
    (re.compile(r'\bModul[üu]bersicht\b', re.IGNORECASE), "Moduluebersicht"),
    (re.compile(r'\bCurriculum\b', re.IGNORECASE), "Studienverlaufsplan"),
]

PROGRAM_CODE_MAP = {
    "inf": "Informatik",
    "winf": "Wirtschaftsinformatik",
    "wing": "Wirtschaftsingenieurwesen",
    "bwl": "Betriebswirtschaftslehre",
    "ecom": "E-Commerce",
    "minf": "Medieninformatik",
    "tinf": "Technische Informatik",
    "ite": "IT-Ingenieurwesen",
    "its": "IT-Sicherheit",
    "stec": "Smart Technology",
    "dsai": "Data Science & Artificial Intelligence",
    "cgt": "Computer Games Technology",
    "awp": "Angewandte Wirtschaftspsychologie & Data Analytics",
    "wim": "Wirtschaftsinformatik - IT-Management",
    "sdbm": "Sustainable & Digital Business Management",
    "itmc": "IT-Management und Consulting",
}

# Für Dokumententyp-spezifische Markdown-Konverter
# "marker": gedacht für marker-pdf
# "pymupdf": Fallback / Standard mit pymupdf4llm
DOC_MARKDOWN_BACKEND: dict[str | None, str] = {
    "Modulübersicht": "marker",
    # "Studienverlaufsplan": "marker",
}

# ========================================
# Hilfsfunktionen
# ========================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def nfc(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")


def detect_doctype(name: str) -> Optional[str]:
    name = nfc(name)
    for rx, label in DOCTYPE_PATTERNS:
        if rx.search(name):
            return label
    return "Sonstiges"


def detect_program_and_degree(name: str) -> Tuple[Optional[str], Optional[str]]:
    name = nfc(name)
    m = PROGCODE_RX.search(name)
    if not m:
        return None, None
    deg_char, code = m.groups()
    degree = "Bachelor" if deg_char.lower() == "b" else "Master" if deg_char.lower() == "m" else None
    program = PROGRAM_CODE_MAP.get(code.lower(), code)
    return degree, program


def detect_version(name: str) -> Optional[str]:
    name = nfc(name)
    m = VERSION_RX.search(name)
    if not m:
        return None
    major, minor, suffix = m.groups()
    return f"{int(major)}.{int(minor)}{suffix.lower()}".rstrip()


def version_key(v: Optional[str]) -> Tuple[int, int, int]:
    if not v:
        return (-1, -1, -1)
    m = VERSION_RX.search(v)
    if not m:
        return (-1, -1, -1)
    major, minor, suffix = m.groups()
    return (int(major), int(minor), ord(suffix.lower()) if suffix else 0)


DATE_RANGE_RX = re.compile(
    r'(\d{1,2})\.(\d{1,2})\.(\d{4})'        # z.B. 1.10.2023
    r'(?:\s*bis\s*(\d{1,2})\.(\d{1,2})\.(\d{4}))?',  # optional "bis" Teil
    re.IGNORECASE
)


def parse_date_range(text: str) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[int]]:
    """
    Text wie "1.10.2023 bis 1.4.2025" -> (start_iso, end_iso, start_year, end_year)
    oder "01.10.2025" -> (start_iso, None, start_year, None)
    """
    text = nfc(text)
    m = DATE_RANGE_RX.search(text)
    if not m:
        return None, None, None, None

    d1, m1, y1, d2, m2, y2 = m.groups()
    start_iso = f"{int(y1):04d}-{int(m1):02d}-{int(d1):02d}"
    start_year = int(y1)

    if d2 and m2 and y2:
        end_iso = f"{int(y2):04d}-{int(m2):02d}-{int(d2):02d}"
        end_year = int(y2)
        if end_iso == start_iso:
            end_iso = None
            end_year = None
    else:
        end_iso = None
        end_year = None

    return start_iso, end_iso, start_year, end_year


# ========================================
# Datendefinition
# ========================================

@dataclass
class Document:
    url: str
    filename: str
    source_page: str                       # Von welcher HTML-Seite stammt der Link
    degree: Optional[str]
    program: Optional[str]
    doctype: Optional[str]
    version: Optional[str]

    start_date: Optional[int] = None       
    end_date: Optional[int] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None

    # aktueller Stand / archiviert
    is_current: Optional[bool] = None
    status: Optional[str] = None

    # Pfade
    local_pdf_path: Optional[str] = None
    local_md_path: Optional[str] = None

    # Zusatzinfos, v.a. von Studiengangsseiten
    html_context: Optional[str] = None     # Textblock um den Link herum


# ========================================
# 1. HTML holen und Tabellen mit Zeiträumen auswerten
# ========================================

def fetch_html(url: str) -> str:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def extract_timeframes_from_pruefungsordnungen(html: str) -> Dict[Tuple[str, str, str], Dict[str, Optional[int | str]]]:
    """
    Liest die Tabellen "Curricula Bachelor/Master" + "Vorherige Curricula" usw. aus
    und bildet eine Map:
        (degree, program, version) -> {start_date, end_date, start_year, end_year}
    Wir nutzen das dann, um die Jahre auf alle Dokumente mit gleicher interner Version
    (z.B. B_Inf23.0) zu übertragen. Die Tabellen sind z.B. mit Spalten
    'Gültig für den Immatrikulationszeitraum' bzw. 'Immatrikulation ab dem'
    aufgebaut. 
    """
    soup = BeautifulSoup(html, "html.parser")
    mapping: Dict[Tuple[str, str, str], Dict[str, Optional[int | str]]] = {}

    for table in soup.select("table.contenttable"):
        # Header-Zeile bestimmen
        header_row = table.find("tr")
        if not header_row:
            continue
        headers = [th.get_text(" ", strip=True) for th in header_row.find_all("th")]
        if not headers:
            continue

        # Finde Spaltenindex für Zeiträume
        try:
            col_idx_range = headers.index("Gültig für den Immatrikulationszeitraum")
        except ValueError:
            col_idx_range = None

        try:
            col_idx_immatrik = headers.index("Immatrikulation ab dem")
        except ValueError:
            col_idx_immatrik = None

        if col_idx_range is None and col_idx_immatrik is None:
            # nicht die Tabellen, die wir brauchen
            continue

        # "Studiengang" Spalte ist meist an Index 0 
        # Uns interessiert aber primär, welche Links (PDFs) in der Zeile liegen.
        for row in table.find_all("tr")[1:]:
            cells = row.find_all("td")
            if not cells:
                continue

            timeframe_cell = None
            if col_idx_range is not None and col_idx_range < len(cells):
                timeframe_cell = cells[col_idx_range]
            elif col_idx_immatrik is not None and col_idx_immatrik < len(cells):
                timeframe_cell = cells[col_idx_immatrik]

            if timeframe_cell:
                timeframe_text = timeframe_cell.get_text(" ", strip=True)
                start_date, end_date, start_year, end_year = parse_date_range(timeframe_text)
            else:
                start_date = end_date = None
                start_year = end_year = None

            # Alle PDF-Links in der Zeile einsammeln und auf Version / Studiengang mappen
            for a in row.select("a[href$='.pdf']"):
                href = a.get("href")
                if not href or not isinstance(href, str):
                    continue
                link_text = a.get_text(" ", strip=True) or os.path.basename(href)
                degree, program = detect_program_and_degree(link_text)
                version = detect_version(link_text) or detect_version(os.path.basename(href))

                if not version:
                    continue

                degree = degree or "Allgemein"
                program = program or "Sonstiges"

                key = (degree, program, version)
                # wenn mehrere Tabellenzeilen die gleiche Version beschreiben,
                # nehmen wir die erste oder überschreiben – je nach Präferenz.
                mapping[key] = {
                    "start_date": start_date,
                    "end_date": end_date,
                    "start_year": start_year,
                    "end_year": end_year,
                }

    return mapping


# ========================================
# 2. PDF-Links aus Prüfungsordnungs-Seite holen
# ========================================

def crawl_pdfs_from_pruefungsordnungen(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    urls: List[str] = []
    for a in soup.select("a[href$='.pdf']"):
        href = a.get("href")
        if not href or not isinstance(href, str):
            continue
        url = urljoin(BASE_URL, href)
        if url not in urls:
            urls.append(url)
    return urls


def classify_pdfs(urls: List[str], source_page: str,
                  timeframe_map: Dict[Tuple[str, str, str], Dict[str, Optional[int]]]
                  ) -> List[Document]:
    docs: List[Document] = []

    for url in urls:
        filename = os.path.basename(url)
        doctype = detect_doctype(filename)
        degree, program = detect_program_and_degree(filename)
        version = detect_version(filename)

        degree = degree or "Allgemein"
        program = program or "Sonstiges"

        start_date = end_date = None
        start_year = end_year = None
        if version:
            tf = timeframe_map.get((degree, program, version))
            if tf:
                start_date = tf["start_date"]
                end_date = tf["end_date"]
                start_year = tf["start_year"]
                end_year = tf["end_year"]

        docs.append(Document(
            url=url,
            filename=filename,
            source_page=source_page,
            degree=degree,
            program=program,
            doctype=doctype,
            version=version,
            start_date=start_date,
            end_date=end_date,
            start_year=start_year,
            end_year=end_year,
        ))
    return docs


# ========================================
# 3. Studiengangsspezifische Seiten verarbeiten
# ========================================

def crawl_program_pages(page_urls: List[str],
                        known_pdf_urls: set[str],
                        timeframe_map: Dict[Tuple[str, str, str], Dict[str, Optional[int]]]
                        ) -> List[Document]:
    """
    Sucht auf einzelnen Studiengangsseiten nach PDFs.
    - Nur Dokumenttypen, die wir auch auf der Prüfungsordnungsseite kennen
      (also via DOCTYPE_PATTERNS != "Sonstiges").
    - Nur solche PDFs, die NICHT schon auf der Prüfungsordnungsseite vorkommen.
    - Zusätzlich wird ein Infotext (HTML-Kontext) um den Link herum extrahiert. 
    """
    docs: List[Document] = []

    for page_url in page_urls:
        try:
            html = fetch_html(page_url)
        except Exception as e:
            print(f"⚠️ Fehler beim Laden der Studiengangsseite {page_url}: {e}")
            continue

        soup = BeautifulSoup(html, "html.parser")

        for a in soup.select("a[href$='.pdf']"):
            href = a.get("href")
            if not href or not isinstance(href, str):
                continue
            url = urljoin(BASE_URL, href)
            if url in known_pdf_urls:
                # Diese Datei gibt es schon auf der Prüfungsordnungsseite
                continue

            filename = os.path.basename(url)
            doctype = detect_doctype(filename)
            if doctype == "Sonstiges":
                # Dokumententypen, die nicht in den DOCTYPE_PATTERNS vorkommen,
                # wollen wir nicht berücksichtigen.
                continue

            degree, program = detect_program_and_degree(filename)
            version = detect_version(filename)

            degree = degree or "Allgemein"
            program = program or "Sonstiges"

            # Kontext-Text suchen (nächster Parentblock mit Text)
            context_node = a.find_parent(["p", "li", "div", "section", "article"])
            context_text = context_node.get_text(" ", strip=True) if context_node else None

            start_date = end_date = None
            start_year = end_year = None
            if version:
                tf = timeframe_map.get((degree, program, version))
                if tf:
                    start_date = tf["start_date"]
                    end_date = tf["end_date"]
                    start_year = tf["start_year"]
                    end_year = tf["end_year"]

            docs.append(Document(
                url=url,
                filename=filename,
                source_page=page_url,
                degree=degree,
                program=program,
                doctype=doctype,
                version=version,
                start_date=start_date,
                end_date=end_date,
                start_year=start_year,
                end_year=end_year,
                html_context=context_text,
            ))
    return docs


# ========================================
# 4. Aktuell/Archiviert bestimmen
# ========================================

def mark_current_versions(docs: List[Document]):
    groups: Dict[Tuple[str, str, str], List[Document]] = {}
    for d in docs:
        key = (d.degree or "?", d.program or "?", d.doctype or "?")
        groups.setdefault(key, []).append(d)

    for _, items in groups.items():
        latest = max(items, key=lambda x: version_key(x.version))
        for d in items:
            d.is_current = (d is latest or d.version == latest.version)
            d.status = "aktuell" if d.is_current else "archiviert"


# ========================================
# 5. Pfade für PDF & Markdown setzen
# ========================================

def assign_paths(docs: List[Document]):
    """
    Legt fest, in welchem Unterordner jeweils das Original-PDF und die Markdown-Datei
    landen sollen. Struktur:
      data/documents/original-docs/<degree>/<program>/<version_status>/<filename>
      data/documents/md-docs/<degree>/<program>/<version_status>/<filename>.md
    """
    for d in docs:
        degree = d.degree or "Allgemein"
        program = d.program or "Sonstiges"
        version = d.version or "unknown"
        status = d.status or "archiviert"
        version_folder = f"{version}_{status}"

        pdf_dir = os.path.join(ORIGINAL_DIR, degree, program, version_folder)
        md_dir = os.path.join(MD_DIR, degree, program, version_folder)
        ensure_dir(pdf_dir)
        ensure_dir(md_dir)

        d.local_pdf_path = os.path.join(pdf_dir, d.filename)
        basename, _ = os.path.splitext(d.filename)
        d.local_md_path = os.path.join(md_dir, f"{basename}.md")


# ========================================
# 6. Asynchroner Download der PDFs
# ========================================

async def _download_single(session: aiohttp.ClientSession, doc: Document, semaphore: asyncio.Semaphore):
    assert doc.local_pdf_path
    try:
        # Vor dem Download prüfen, ob Datei schon existiert
        if doc.local_pdf_path and os.path.exists(doc.local_pdf_path):
            return


        async with semaphore:
            async with session.get(doc.url, timeout=30) as resp:
                if resp.status != 200:
                    print(f"Fehler beim Laden {doc.url}: HTTP {resp.status}")
                    return
                data = await resp.read()

            ensure_dir(os.path.dirname(doc.local_pdf_path))
            async with aiofiles.open(doc.local_pdf_path, "wb") as f:
                await f.write(data)

    except Exception as e:
        print(f"Async-Download-Fehler bei {doc.url}: {e}")



async def _download_pdfs_async(docs: List[Document], max_concurrent: int = 5):

    ssl_context = ssl.create_default_context(cafile=certifi.where())

    connector = aiohttp.TCPConnector(
        limit=max_concurrent,
        ssl=ssl_context,   # WICHTIG: nutzt jetzt certifi-CA statt nur Windows-Store
    )
    timeout = aiohttp.ClientTimeout(total=60)
    semaphore = asyncio.Semaphore(max_concurrent)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [_download_single(session, d, semaphore) for d in docs]

        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Lade PDFs herunter"):
            try:
                await f
            except Exception as e:
                # Fängt z.B. SSL-Fehler eines einzelnen Downloads ab
                print(f"Fehler im Async-Task: {e}")



def download_pdfs(docs: List[Document]):
    """
    Wrapper um den asynchronen Download.
    """
    ensure_dir(ORIGINAL_DIR)
    
    asyncio.run(_download_pdfs_async(docs))


# ========================================
# 7. PDF -> Markdown
# ========================================

def pdf_to_markdown_pymupdf(pdf_path: str) -> str:
    if not pymupdf4llm:
        raise RuntimeError(
            "pymupdf4llm ist nicht installiert. Bitte 'pip install pymupdf4llm' ausführen."
        )
    return pymupdf4llm.to_markdown(pdf_path)


def pdf_to_markdown_marker(pdf_path: str) -> str:
    """
    """
    # TODO: hier marker-pdf integrieren
    # vorerst Fallback auf pymupdf
    return pdf_to_markdown_pymupdf(pdf_path)


def convert_pdf_to_markdown(doc: Document):
    if not doc.local_pdf_path or not os.path.exists(doc.local_pdf_path):
        return
    if not doc.local_md_path:
        return

    # Falls Markdown bereits existiert, nicht nochmal konvertieren
    if os.path.exists(doc.local_md_path):
        return

    backend = DOC_MARKDOWN_BACKEND.get(doc.doctype, "pymupdf")

    try:
        if backend == "marker":
            md = pdf_to_markdown_marker(doc.local_pdf_path)
        else:
            md = pdf_to_markdown_pymupdf(doc.local_pdf_path)
    except Exception as e:
        print(f"⚠️ Fehler bei Markdown-Konvertierung ({backend}) von {doc.local_pdf_path}: {e}")
        return

    ensure_dir(os.path.dirname(doc.local_md_path))
    with open(doc.local_md_path, "w", encoding="utf-8") as f:
        f.write(md)


def convert_all_to_markdown(docs: List[Document]):
    for d in tqdm(docs, desc="📝 Konvertiere PDFs nach Markdown"):
        convert_pdf_to_markdown(d)


# ========================================
# 8. Metadaten schreiben
# ========================================

def write_outputs(docs: List[Document]):
    docs_sorted = sorted(docs, key=lambda x: (
        x.degree or "",
        x.program or "",
        version_key(x.version),
        x.doctype or "",
    ))

    ensure_dir(DATA_BASE_DIR)

    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump([asdict(d) for d in docs_sorted], f, indent=2, ensure_ascii=False)

    with open(META_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "url", "filename", "source_page",
            "degree", "program", "doctype", "version",
            "start_date", "end_date", "start_year", "end_year",
            "status",
            "local_pdf_path", "local_md_path",
            "html_context",
        ])
        for d in docs_sorted:
            writer.writerow([
                d.url, d.filename, d.source_page,
                d.degree, d.program, d.doctype, d.version,
                d.start_date, d.end_date, d.start_year, d.end_year,
                d.status,
                d.local_pdf_path, d.local_md_path,
                d.html_context,
            ])


# ========================================
# 9. Hauptablauf
# ========================================

def main():
    ensure_dir(DATA_BASE_DIR)

    # 1) Prüfungsordnungsseite laden
    print(f"🔍 Lade Prüfungsordnungsseite: {PRUEFUNGSORDNUNGEN_URL}")
    html_pruef = fetch_html(PRUEFUNGSORDNUNGEN_URL)

    # 2) Zeiträume aus Tabellen extrahieren (Immatrik./Gültigkeitszeiträume) 
    timeframe_map = extract_timeframes_from_pruefungsordnungen(html_pruef)

    # 3) Alle PDFs von der Prüfungsordnungsseite holen
    pdf_urls_main = crawl_pdfs_from_pruefungsordnungen(html_pruef)
    print(f"✅ {len(pdf_urls_main)} PDF-Links auf der Prüfungsordnungsseite gefunden.")

    docs_main = classify_pdfs(pdf_urls_main, PRUEFUNGSORDNUNGEN_URL, timeframe_map)

    # 4) Studiengangsspezifische Seiten: nur zusätzliche PDFs, nur bekannte DOCTYPES
    additional_docs: List[Document] = []
    if PROGRAM_PAGE_URLS:
        known_set = set(pdf_urls_main)
        docs_program = crawl_program_pages(PROGRAM_PAGE_URLS, known_set, timeframe_map)
        print(f"➕ {len(docs_program)} zusätzliche Programm-spezifische PDFs gefunden.")
        additional_docs.extend(docs_program)

    all_docs = docs_main + additional_docs

    # 5) aktuell/archiviert bestimmen
    mark_current_versions(all_docs)

    # 6) Pfade für PDF & Markdown festlegen
    assign_paths(all_docs)

    # 7) PDFs herunterladen (asynchron, falls aiohttp verfügbar)
    download_pdfs(all_docs)

    # 8) Alle PDFs nach Markdown konvertieren
    convert_all_to_markdown(all_docs)

    # 9) Metadaten schreiben
    write_outputs(all_docs)

    print(
        f"\n🎉 Fertig! {len(all_docs)} Dokumente klassifiziert, "
        f"heruntergeladen und in '{DATA_BASE_DIR}' (original-docs, md-docs, meta.json/csv) gespeichert."
    )


if __name__ == "__main__":
    main()
