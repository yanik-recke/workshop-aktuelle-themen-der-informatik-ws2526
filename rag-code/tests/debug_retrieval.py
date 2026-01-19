"""Debug script to test retrieval directly."""
import os
import sys
import os.path as op

# Load .env
_RAG_ROOT = op.dirname(op.dirname(op.abspath(__file__)))
sys.path.insert(0, _RAG_ROOT)

from dotenv import load_dotenv
load_dotenv(op.join(_RAG_ROOT, ".env"), override=True)

print(f"OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL')}")
print(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')[:10]}...")

# Test the full eval_adapter flow
print("\n=== Testing eval_adapter.run_single_turn ===")
from eval_adapter import run_single_turn, retrieve_chunks
query = "Was ist die Regelstudienzeit im Bachelor Informatik?"
try:
    answer, chunks = run_single_turn(query)
    print(f"Answer: {answer[:200]}...")
    print(f"Chunks returned: {len(chunks)}")
except Exception as e:
    print(f"run_single_turn FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Testing eval_adapter.retrieve_chunks ===")
try:
    chunks = retrieve_chunks(query)
    print(f"Chunks returned: {len(chunks)}")
    for i, c in enumerate(chunks[:2]):
        print(f"  [{i+1}] {c[:100]}...")
except Exception as e:
    print(f"retrieve_chunks FAILED: {e}")
    import traceback
    traceback.print_exc()

from hybrid_retrieval import hybrid_search, semantic_search, keyword_search
from document_store import get_document_store

# Test 1: Check document store
print("\n=== Document Store ===")
store = get_document_store()
all_docs = store.filter_documents()
print(f"Total documents in store: {len(all_docs)}")
if all_docs:
    print(f"Sample doc: {all_docs[0].meta.get('filename', '?')}")

# Test 2: Keyword search only
print("\n=== Keyword Search ===")
query = "Regelstudienzeit Bachelor Informatik"
kw_docs = keyword_search(query, top_k=3)
print(f"Keyword search found: {len(kw_docs)} docs")
for d in kw_docs[:2]:
    print(f"  - {d.meta.get('filename', '?')}: {d.content[:80]}...")

# Test 3: Semantic search
print("\n=== Semantic Search ===")
try:
    sem_docs = semantic_search(query, top_k=3)
    print(f"Semantic search found: {len(sem_docs)} docs")
    for d in sem_docs[:2]:
        print(f"  - {d.meta.get('filename', '?')}: {d.content[:80]}...")
except Exception as e:
    print(f"Semantic search FAILED: {e}")

# Test 4: Hybrid search
print("\n=== Hybrid Search ===")
try:
    hyb_docs = hybrid_search(query, top_k=3)
    print(f"Hybrid search found: {len(hyb_docs)} docs")
    for d in hyb_docs[:2]:
        print(f"  - {d.meta.get('filename', '?')}: {d.content[:80]}...")
except Exception as e:
    print(f"Hybrid search FAILED: {e}")
