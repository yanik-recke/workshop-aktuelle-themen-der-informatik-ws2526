
from typing import Dict

from haystack import Pipeline
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever

from config import OPENAI_CHAT_MODEL, OPENAI_EMBED_MODEL, TOP_K, MAX_CONTEXT_TOKENS
from document_store import get_document_store

import tiktoken



def _truncate_text(text: str, max_tokens: int) -> str:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    if len(tokens) <= max_tokens:
        return text

    truncated = enc.decode(tokens[:max_tokens])
    return truncated + "\n...[Content truncated for length]..."


def _truncate_documents(documents, max_tokens: int):
    """
    Shrink each retrieved document to max_tokens.
    """
    truncated = []
    for doc in documents:
        doc.content = _truncate_text(doc.content, max_tokens)
        truncated.append(doc)
    return truncated




PROMPT_TEMPLATE = """
You are a helpful FH-Wedel assistant.
Answer **only** based on the retrieved documents and the memory summary.
If the answer is not clearly contained in the documents, say "I don't know".

### Memory Summary
{{ memory_summary or "None" }}

### Conversation
{{ conversation_history or "None" }}

### Retrieved Documents
{% if documents %}
{% for doc in documents %}
📘 {{ doc.meta.get("doctype", "") }} - {{ doc.meta.get("program", "") }} - {{ doc.meta.get("degree", "") }}
{{ doc.content }}
---
{% endfor %}
{% else %}
(no documents found)
{% endif %}

### Question
{{ query }}

Give a precise answer based strictly on the documents.
"""



def create_rag_pipeline() -> Pipeline:
    document_store = get_document_store()

    text_embedder = OpenAITextEmbedder(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        model=OPENAI_EMBED_MODEL,
    )

    retriever = QdrantEmbeddingRetriever(
        document_store=document_store,
        top_k=TOP_K,
        return_embedding=False,
    )

    prompt_builder = PromptBuilder(template=PROMPT_TEMPLATE)

    generator = OpenAIGenerator(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        model=OPENAI_CHAT_MODEL,
        generation_kwargs={"temperature": 0.3},
    )

    pipeline = Pipeline()
    pipeline.add_component("text_embedder", text_embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("generator", generator)

    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder", "generator")

    return pipeline



def run_rag_query(
    pipeline: Pipeline,
    *,
    original_query: str,
    expanded_query: str,
    memory_summary: str,
    conversation_history: str,
    filters: Dict = None,   # <-- optional gemacht
) -> str:
    """
    Führt eine RAG-Abfrage aus und gibt die generierte Antwort zurück.
    """

    inputs = {
        "text_embedder": {"text": expanded_query},
        "retriever": {},
        "prompt_builder": {
            "query": original_query,
            "memory_summary": memory_summary,
            "conversation_history": conversation_history,
        },
        "generator": {},
    }

    # Nur falls wieder aktiv
    if filters:
        inputs["retriever"]["filters"] = filters

    result = pipeline.run(inputs)
    replies = result["generator"]["replies"]
    if not replies:
        return "No reply: I don't know."
    return replies[0].strip()


def build_filters_from_context(context_state: Dict) -> Dict:
    """
    Temporäre Dummy-Funktion.
    """
    return {}