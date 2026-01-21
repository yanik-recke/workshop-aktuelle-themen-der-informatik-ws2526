from typing import Optional
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from config import EMBEDDING_DIM, QDRANT_COLLECTION, QDRANT_URL

_document_store: Optional[QdrantDocumentStore] = None

def get_document_store() -> QdrantDocumentStore:
    global _document_store
    if _document_store is None:
        _document_store = QdrantDocumentStore(
            url=QDRANT_URL,  # Connect to Qdrant server instead of local file
            index=QDRANT_COLLECTION,
            embedding_dim=EMBEDDING_DIM,
            recreate_index=False  # Data already indexed
        )
    return _document_store
