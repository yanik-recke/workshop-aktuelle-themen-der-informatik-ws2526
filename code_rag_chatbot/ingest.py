import os
import re
import orjson
import fitz
import hashlib
import time
from tqdm import tqdm
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from config import *
from utils import sanitize_key

import tiktoken  

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

os.makedirs(EMB_DIR, exist_ok=True)
client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL)

# Choose tokenizer encoding for the embedding model
TOKENIZER = tiktoken.get_encoding("cl100k_base")
 # adjust if your model uses different encoding

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def file_hash(text):
    return hashlib.sha1(text.strip().lower().encode("utf-8")).hexdigest()

def read_pdf_text(path):
    """Read text from a PDF file."""
    text = []
    try:
        with fitz.open(path) as doc:
            for p in doc:
                txt = p.get_text("text")
                if txt:
                    text.append(txt)
    except Exception as e:
        print(f"⚠️ Fehler beim Lesen {path}: {e}")
    return "\n".join(text)

def count_tokens(text: str) -> int:
    """Return number of tokens in text using the tokenizer."""
    return len(TOKENIZER.encode(text))

def extract_paragraphs(text: str):
    """Split text into paragraphs and detect headings."""
    text = re.sub(r"\n{2,}", "\n\n", text.strip())
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    paragraphs = []
    current_heading = ""
    for para in raw_paragraphs:
        is_heading = (
            len(para) < 80 and (
                para.endswith(":") or 
                para.isupper() or 
                re.match(r"^[A-Z][A-Za-z\s]+$", para)
            )
        )
        if is_heading:
            current_heading = para
        else:
            paragraphs.append((current_heading, para))
    return paragraphs

def chunk_text_by_tokens(text: str, max_tokens: int = CHUNK_SIZE, overlap_tokens: int = CHUNK_OVERLAP):
    """Chunk text dynamically by tokens, preserving paragraph boundaries and headings."""
    paragraphs = extract_paragraphs(text)
    chunks = []
    current_chunk_parts = []
    current_chunk_tokens = 0
    last_para_text = ""

    for heading, para in paragraphs:
        # Prepend heading if exists
        if heading:
            para_text = f"{heading}\n{para}"
        else:
            para_text = para
        para_tokens = count_tokens(para_text)

        # If adding this paragraph would exceed max_tokens, flush current chunk
        if current_chunk_tokens + para_tokens > max_tokens and current_chunk_parts:
            chunk = "\n\n".join(current_chunk_parts).strip()
            chunks.append(chunk)

            # For overlap: include last paragraph text as new start
            # (we could use last_para_text's last overlap_tokens if desired)
            current_chunk_parts = [last_para_text] if last_para_text else []
            current_chunk_tokens = count_tokens(last_para_text) if last_para_text else 0
        # Append paragraph
        current_chunk_parts.append(para_text)
        current_chunk_tokens += para_tokens
        last_para_text = para_text

    # Add leftover
    if current_chunk_parts:
        chunk = "\n\n".join(current_chunk_parts).strip()
        chunks.append(chunk)

    return chunks

def embed_texts(texts):
    """Embed text batch."""
    resp = client.embeddings.create(
        input=[t for t in texts],
        model=EMBED_MODEL
    )
    return [d.embedding for d in resp.data]

def ensure_collection(dimensions: int):
    """Create Qdrant collection if not exists."""
    collections = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in collections:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=rest.VectorParams(size=dimensions, distance=rest.Distance.COSINE)
        )

# ---------------------------------------------------------
# ONE-HOT HELPERS
# ---------------------------------------------------------
def collect_unique_values(metadata):
    degrees = sorted({m.get("degree") for m in metadata if m.get("degree")})
    programs = sorted({m.get("program") for m in metadata if m.get("program")})
    doctypes = sorted({m.get("doctype") for m in metadata if m.get("doctype")})
    statuses = sorted({m.get("status") for m in metadata if m.get("status")})
    return {"degrees": degrees, "programs": programs, "doctypes": doctypes, "statuses": statuses}

# ---------------------------------------------------------
# CACHE MANAGEMENT
# ---------------------------------------------------------
def load_embedding_cache():
    if os.path.exists(EMB_CACHE_FILE):
        with open(EMB_CACHE_FILE, "rb") as f:
            return orjson.loads(f.read())
    return {}

def save_embedding_cache(cache):
    data = orjson.dumps(cache)
    with open(EMB_CACHE_FILE, "wb") as f:
        f.write(data)

def uploaded_ids():
    """Get already stored point IDs from Qdrant."""
    try:
        points, _ = qdrant.scroll(collection_name=COLLECTION_NAME, limit=1)
        return {p.id for p in points}
    except Exception:
        return set()

# ---------------------------------------------------------
# MAIN INGEST FUNCTION
# ---------------------------------------------------------
def ingest_documents():
    # --- Load metadata ---
    with open(META_FILE, "rb") as f:
        meta = orjson.loads(f.read())

    value_sets = collect_unique_values(meta)
    chunk_map = {}
    emb_cache = load_embedding_cache()
    print(f"🗂️ Embedding Cache enthält {len(emb_cache)} gespeicherte Einträge.")

    # --- Extract and chunk PDFs ---
    for d in tqdm(meta, desc="📄 Analysiere PDFs"):
        path = d.get("local_path") or d.get("filename")
        if not path or not os.path.exists(path):
            continue

        text = read_pdf_text(path)
        if not text.strip():
            continue

        # Use token-based chunking
        for chunk in chunk_text_by_tokens(text, max_tokens=CHUNK_SIZE, overlap_tokens=CHUNK_OVERLAP):
            h = file_hash(chunk)
            if h not in chunk_map:
                chunk_map[h] = {"text": chunk, "combinations": set()}
            combo = (d.get("degree"), d.get("program"), d.get("doctype"), d.get("version"), d.get("status"))
            chunk_map[h]["combinations"].add(combo)

    print(f"✅ {len(chunk_map)} eindeutige Chunks erkannt.")

    # --- Compute embeddings (resume if possible) ---
    all_chunks = list(chunk_map.items())
    embeddings = {}

    new_texts = []
    new_hashes = []

    for h, c in all_chunks:
        if h in emb_cache:
            embeddings[h] = emb_cache[h]
        else:
            new_texts.append(c["text"])
            new_hashes.append(h)

    print(f"🧠 {len(new_texts)} neue Embeddings werden berechnet...")

    for i in tqdm(range(0, len(new_texts), EMB_BATCH), desc="🔢 Berechne Embeddings"):
        batch = new_texts[i:i+EMB_BATCH]
        batch_embeds = embed_texts(batch)
        for h, emb in zip(new_hashes[i:i+EMB_BATCH], batch_embeds):
            embeddings[h] = emb
            emb_cache[h] = emb
        save_embedding_cache(emb_cache)
        time.sleep(0.05)

    print(f"💾 Embedding-Cache aktualisiert ({len(embeddings)} total).")

    # --- Upload points (resumable) ---
    print("📦 Lade Chunks in Qdrant...")
    # ensure collection with right dimension
    first_h = all_chunks[0][0]
    ensure_collection(len(embeddings[first_h]))
    existing_ids = uploaded_ids()
    print(f"🔍 {len(existing_ids)} Punkte bereits in Qdrant vorhanden.")

    points = []
    counter = 0
    for i, (h, c) in enumerate(all_chunks):
        if str(i) in existing_ids:
            continue

        emb = embeddings[h]
        all_metas = c["combinations"]

        meta_dict = {"degree": set(), "program": set(), "doctype": set(), "status": set()}
        for deg, prog, doc, ver, stat in all_metas:
            if deg: meta_dict["degree"].add(deg)
            if prog: meta_dict["program"].add(prog)
            if doc: meta_dict["doctype"].add(doc)
            if stat: meta_dict["status"].add(stat)

        payload = {
            "text": c["text"],
            "combos": [list(x) for x in all_metas],
            "degree_onehot": {sanitize_key(k): 1 if k in meta_dict["degree"] else 0 for k in value_sets["degrees"]},
            "program_onehot": {sanitize_key(k): 1 if k in meta_dict["program"] else 0 for k in value_sets["programs"]},
            "doctype_onehot": {sanitize_key(k): 1 if k in meta_dict["doctype"] else 0 for k in value_sets["doctypes"]},
            "status_onehot": {sanitize_key(k): 1 if k in meta_dict["status"] else 0 for k in value_sets["statuses"]},
        }
        points.append(rest.PointStruct(id=i, vector=emb, payload=payload))

        if len(points) >= UPLOAD_BATCH:
            try:
                qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
                counter += len(points)
            except Exception as e:
                print(f"⚠️ Fehler beim Upload: {e}")
            points = []
            time.sleep(0.2)

    # --- Upload remaining ---
    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        counter += len(points)

    print(f"🎉 Ingest abgeschlossen! {counter} neue Punkte hinzugefügt.")
    save_embedding_cache(emb_cache)

# ---------------------------------------------------------
if __name__ == "__main__":
    ingest_documents()
