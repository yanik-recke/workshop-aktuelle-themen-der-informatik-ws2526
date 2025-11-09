import os
import re
import csv
import json
import time
import unicodedata
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple
from tqdm import tqdm

# ========================================
# Konfiguration
# ========================================
BASE_URL = "https://www.fh-wedel.de"
START_URL = "https://www.fh-wedel.de/studieren/pruefungscenter/pruefungsordnungen/"
DOWNLOAD_DIR = "data_fh_wedel"
META_JSON = "fhwedel_docs.json"
META_CSV = "fhwedel_docs.csv"

# ========================================
# Musterdefinitionen
# ========================================
PROGCODE_RX = re.compile(
    r'(?<![A-Za-z0-9])([BM])[_\- ]([A-ZÄÖÜa-zäöü]{2,12})(?=\d|_|\.|$)',
    re.UNICODE
)
VERSION_RX = re.compile(r'(\d{2})\.(\d{1,2})([a-zA-Z]?)')
DOCTYPE_PATTERNS = [
    (re.compile(r'\bSPO\b', re.IGNORECASE), "Studien- und Prüfungsordnung"),
    (re.compile(r'\bModulhandbuch\b', re.IGNORECASE), "Modulhandbuch"),
    (re.compile(r'\bModul[üu]bersicht\b', re.IGNORECASE), "Modulübersicht"),
    (re.compile(r'\bCurriculum\b', re.IGNORECASE), "Studienverlaufsplan"),
]

PROGRAM_CODE_MAP = {
    "Inf": "Informatik",
    "WInf": "Wirtschaftsinformatik",
    "WIng": "Wirtschaftsingenieurwesen",
    "BWL": "Betriebswirtschaftslehre",
    "ECom": "E-Commerce",
    "MInf": "Medieninformatik",
    "TInf": "Technische Informatik",
    "ITE": "IT-Ingenieurwesen",
    "ITS": "IT-Sicherheit",
    "STec": "Smart Technology",
    "DSAI": "Data Science & Artificial Intelligence",
    "CGT": "Computer Games Technology",
    "AWP": "Angewandte Wirtschaftspsychologie & Data Analytics",
    "WIM": "Wirtschaftsinformatik - IT-Management",
    "SDBM": "Sustainable & Digital Business Management",
    "ITMC": "IT-Management und Consulting",
}

# ========================================
# Hilfsfunktionen
# ========================================
def nfc(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")

def detect_doctype(filename: str) -> Optional[str]:
    for rx, label in DOCTYPE_PATTERNS:
        if rx.search(filename):
            return label
    return "Sonstiges"

def detect_program_and_degree(filename: str) -> Tuple[Optional[str], Optional[str]]:
    m = PROGCODE_RX.search(filename)
    if not m:
        return None, None
    deg_char, code = m.groups()
    degree = "Bachelor" if deg_char == "B" else "Master" if deg_char == "M" else None
    program = PROGRAM_CODE_MAP.get(code, code)
    return degree, program

def detect_version(filename: str) -> Optional[str]:
    m = VERSION_RX.search(filename)
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

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ========================================
# Datendefinition
# ========================================
@dataclass
class Document:
    url: str
    filename: str
    degree: Optional[str]
    program: Optional[str]
    doctype: Optional[str]
    version: Optional[str]
    is_current: Optional[bool] = None
    status: Optional[str] = None
    local_path: Optional[str] = None

# ========================================
# Crawler
# ========================================
def crawl_pdfs(start_url: str) -> List[str]:
    print(f"🔍 Lade Webseite: {start_url}")
    resp = requests.get(start_url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    pdf_urls = []
    for a in soup.select("a[href$='.pdf']"):
        href = a.get("href")
        if not href:
            continue
        url = urljoin(BASE_URL, href)
        if url not in pdf_urls:
            pdf_urls.append(url)
    print(f"✅ {len(pdf_urls)} PDF-Links gefunden.")
    return pdf_urls

# ========================================
# Analyse der Dateinamen
# ========================================
def classify_pdfs(urls: List[str]) -> List[Document]:
    docs = []
    for url in urls:
        filename = os.path.basename(url)
        doctype = detect_doctype(filename)
        degree, program = detect_program_and_degree(filename)
        version = detect_version(filename)

        docs.append(Document(
            url=url,
            filename=filename,
            degree=degree or "Allgemein",
            program=program or "Sonstiges",
            doctype=doctype,
            version=version
        ))
    return docs

# ========================================
# Aktuell/Archiviert bestimmen
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
# Download
# ========================================
def download_pdfs(docs: List[Document]):
    ensure_dir(DOWNLOAD_DIR)
    for d in tqdm(docs, desc="📥 Lade PDFs herunter"):
        degree = d.degree or "Allgemein"
        program = d.program or "Sonstiges"
        version = d.version or "unknown"
        status = d.status or "archiviert"

        subdir = os.path.join(DOWNLOAD_DIR, degree, program, f"{version}_{status}")
        ensure_dir(subdir)
        local_path = os.path.join(subdir, d.filename)
        d.local_path = local_path

        if os.path.exists(local_path):
            continue  # nur neue laden

        try:
            r = requests.get(d.url, timeout=30)
            if r.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(r.content)
                time.sleep(0.2)
            else:
                print(f"⚠️ Fehler beim Laden: {d.url} ({r.status_code})")
        except Exception as e:
            print(f"⚠️ Fehler beim Download {d.url}: {e}")

# ========================================
# Metadaten speichern
# ========================================
def write_outputs(docs: List[Document]):
    docs_sorted = sorted(docs, key=lambda x: (
        x.degree or "",
        x.program or "",
        version_key(x.version)
    ))

    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump([asdict(d) for d in docs_sorted], f, indent=2, ensure_ascii=False)

    with open(META_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "filename", "degree", "program", "doctype", "version", "status", "local_path"])
        for d in docs_sorted:
            writer.writerow([d.url, d.filename, d.degree, d.program, d.doctype, d.version, d.status, d.local_path])

# ========================================
# Hauptablauf
# ========================================
def main():
    pdf_urls = crawl_pdfs(START_URL)
    docs = classify_pdfs(pdf_urls)
    mark_current_versions(docs)
    download_pdfs(docs)
    write_outputs(docs)
    print(f"\n🎉 Fertig! {len(docs)} Dokumente klassifiziert und gespeichert in '{DOWNLOAD_DIR}'.")

if __name__ == "__main__":
    main()
