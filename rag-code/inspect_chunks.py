"""
Tool to inspect parsed chunks from specific files in the Qdrant database.
Run: python inspect_chunks.py <filename_pattern>
Example: python inspect_chunks.py Curriculum_B_Inf
"""
import sys
from document_store import get_document_store


def inspect_chunks(filename_pattern: str = None, max_chunks: int = 50):
    """
    Inspect chunks stored in Qdrant, optionally filtering by filename.
    """
    store = get_document_store()
    all_docs = store.filter_documents()
    
    print(f"Total documents in store: {len(all_docs)}")
    
    if filename_pattern:
        # Filter by filename
        matching = [
            d for d in all_docs 
            if filename_pattern.lower() in (d.meta.get("filename", "") or "").lower()
        ]
        print(f"Documents matching '{filename_pattern}': {len(matching)}")
    else:
        matching = all_docs
    
    # Group by filename
    by_file = {}
    for doc in matching:
        fname = doc.meta.get("filename", "unknown")
        if fname not in by_file:
            by_file[fname] = []
        by_file[fname].append(doc)
    
    print(f"\nFiles found: {len(by_file)}")
    for fname, docs in sorted(by_file.items()):
        print(f"  - {fname}: {len(docs)} chunks")
    
    # Show chunk details
    print(f"\n{'='*80}")
    print("CHUNK DETAILS")
    print(f"{'='*80}")
    
    shown = 0
    for fname, docs in sorted(by_file.items()):
        if shown >= max_chunks:
            break
        print(f"\n### {fname} ({len(docs)} chunks) ###\n")
        for i, doc in enumerate(docs[:10]):  # Show max 10 per file
            if shown >= max_chunks:
                break
            meta = doc.meta or {}
            print(f"[Chunk {i+1}]")
            print(f"  Program: {meta.get('program', '?')}")
            print(f"  Degree: {meta.get('degree', '?')}")
            print(f"  Doctype: {meta.get('doctype', '?')}")
            print(f"  Module: {meta.get('module_name', '?')}")
            print(f"  Content length: {len(doc.content)} chars")
            print(f"  Preview: {doc.content[:300]}...")
            print("-" * 40)
            shown += 1
        
        if len(docs) > 10:
            print(f"  ... and {len(docs) - 10} more chunks")


def list_all_files():
    """List all unique filenames in the store."""
    store = get_document_store()
    all_docs = store.filter_documents()
    
    filenames = set()
    for doc in all_docs:
        fname = doc.meta.get("filename", "unknown")
        filenames.add(fname)
    
    print(f"Total files indexed: {len(filenames)}")
    for fname in sorted(filenames):
        print(f"  - {fname}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list":
            list_all_files()
        else:
            inspect_chunks(sys.argv[1])
    else:
        print("Usage:")
        print("  python inspect_chunks.py <filename_pattern>  - Inspect chunks for files matching pattern")
        print("  python inspect_chunks.py --list              - List all indexed files")
        print("\nExamples:")
        print("  python inspect_chunks.py Curriculum_B_Inf")
        print("  python inspect_chunks.py Modulhandbuch")
