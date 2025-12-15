from typing import Dict, List

from haystack import Pipeline
from haystack.dataclasses import Document
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
Du bist ein hilfreicher Assistent für die FH Wedel.
Beantworte die Frage **nur** basierend auf den bereitgestellten Dokumenten.
Wenn die Antwort nicht in den Dokumenten enthalten ist, sage "Das kann ich anhand der vorliegenden Dokumente nicht beantworten."
Antworte auf Deutsch, es sei denn die Frage ist auf Englisch gestellt.

### Bisheriger Kontext
{{ memory_summary or "Keiner" }}

### Dokumente
{% if documents %}
{% for doc in documents %}
[Dokument {{ loop.index }}] {{ doc.meta.get("doctype", "") }} | {{ doc.meta.get("program", "") }} | {{ doc.meta.get("degree", "") }}{% if doc.meta.get("comparison_entity") %} | FÜR: {{ doc.meta.get("comparison_entity") }}{% endif %}
{{ doc.content[:1200] }}
{% endfor %}
{% else %}
(Keine Dokumente gefunden)
{% endif %}

### Frage
{{ query }}

### Antwort
"""

COMPARISON_PROMPT_TEMPLATE = """
Du bist ein hilfreicher Assistent für die FH Wedel.
Der Nutzer möchte einen Vergleich zwischen zwei Dingen.
Vergleiche die Dokumente für "{{ entity1 }}" mit denen für "{{ entity2 }}".

Strukturiere deine Antwort wie folgt:
1. Kurze Beschreibung von {{ entity1 }}
2. Kurze Beschreibung von {{ entity2 }}
3. Wichtige Unterschiede
4. Gemeinsamkeiten (falls relevant)

Basiere deine Antwort NUR auf den bereitgestellten Dokumenten.

### Dokumente
{% for doc in documents %}
[{{ doc.meta.get("comparison_entity", "?") }}] {{ doc.meta.get("doctype", "") }} | {{ doc.meta.get("program", "") }}
{{ doc.content[:1000] }}
{% endfor %}

### Frage
{{ query }}

### Vergleich
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
        generation_kwargs={"temperature": 0.3, "max_tokens": 512},
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
    filters: Dict = None,
    use_hybrid: bool = True,
    keywords: List[str] = None,
) -> str:
    """
    Führt eine RAG-Abfrage aus und gibt die generierte Antwort zurück.
    # todo add history again
    Uses hybrid search (keyword + semantic) by default for better retrieval.
    """
    from hybrid_retrieval import hybrid_search
    
    if use_hybrid:
        # Use hybrid search for better keyword + semantic matching
        # Pass enhanced keywords for keyword matching, expanded_query for semantics
        documents = hybrid_search(
            query=expanded_query,
            top_k=TOP_K,
            filters=filters,
            original_query=original_query,
            keywords=keywords,
        )
        
        # Build prompt manually and run generator
        prompt_builder = PromptBuilder(template=PROMPT_TEMPLATE)
        prompt_result = prompt_builder.run(
            documents=documents,
            query=original_query,
            memory_summary=memory_summary,
        )
        
        generator = OpenAIGenerator(
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            model=OPENAI_CHAT_MODEL,
            generation_kwargs={"temperature": 0.3, "max_tokens": 512},
        )
        
        gen_result = generator.run(prompt=prompt_result["prompt"])
        replies = gen_result.get("replies", [])
    else:
        # Original pipeline-based approach
        inputs = {
            "text_embedder": {"text": expanded_query},
            "retriever": {},
            "prompt_builder": {
                "query": original_query,
                "memory_summary": memory_summary,
            },
            "generator": {},
        }
        if filters:
            inputs["retriever"]["filters"] = filters
        
        result = pipeline.run(inputs)
        replies = result["generator"]["replies"]
    
    if not replies:
        return "No reply: I don't know."
    return replies[0].strip()


def run_comparison_query(
    original_query: str,
    documents: List[Document],
    entity1: str,
    entity2: str,
    memory_summary: str = "",
) -> str:
    """
    Run a comparison query with pre-retrieved documents for two entities.
    """
    prompt_builder = PromptBuilder(template=COMPARISON_PROMPT_TEMPLATE)
    prompt_result = prompt_builder.run(
        documents=documents,
        query=original_query,
        entity1=entity1,
        entity2=entity2,
    )
    
    generator = OpenAIGenerator(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        model=OPENAI_CHAT_MODEL,
        generation_kwargs={"temperature": 0.3, "max_tokens": 800},  # More tokens for comparisons
    )
    
    gen_result = generator.run(prompt=prompt_result["prompt"])
    replies = gen_result.get("replies", [])
    
    if not replies:
        return "Ich konnte keinen Vergleich erstellen."
    return replies[0].strip()


def build_filters_from_context(context_state: Dict) -> Dict:
    """
    Temporäre Dummy-Funktion.
    """
    return {}