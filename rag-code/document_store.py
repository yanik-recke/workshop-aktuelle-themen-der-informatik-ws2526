from typing import Optional
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from config import EMBEDDING_DIM, QDRANT_COLLECTION
_document_store: Optional[QdrantDocumentStore] = None

def get_document_store() -> QdrantDocumentStore:
    global _document_store
    if _document_store is None:
        _document_store = QdrantDocumentStore(
            path="qdrant_data",
            index=QDRANT_COLLECTION,
            embedding_dim=EMBEDDING_DIM,      # oder dein EMBEDDING_DIM
            recreate_index=False       # Reindexing with all PDFs 
        )
    return _document_store
