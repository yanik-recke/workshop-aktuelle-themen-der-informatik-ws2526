from typing import Optional
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from config import EMBEDDING_DIM
_document_store: Optional[QdrantDocumentStore] = None

def get_document_store() -> QdrantDocumentStore:
    global _document_store
    if _document_store is None:
        _document_store = QdrantDocumentStore(
            path="qdrant_data",
            index="documents",
            embedding_dim=1024,      # oder dein EMBEDDING_DIM
            recreate_index=False     # nur beim ersten Lauf!
        )
    return _document_store