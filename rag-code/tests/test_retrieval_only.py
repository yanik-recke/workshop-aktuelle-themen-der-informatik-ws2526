import os
import sys
import os.path as op

# Load .env from rag-code folder
_RAG_ROOT = op.dirname(op.dirname(op.abspath(__file__)))
try:
    from dotenv import load_dotenv
    load_dotenv(op.join(_RAG_ROOT, ".env"), override=True)
except ImportError:
    pass

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRelevancyMetric

# Make the rag-code folder importable for eval_adapter
_RAG_ROOT = op.dirname(op.dirname(__file__))
if _RAG_ROOT not in sys.path:
    sys.path.insert(0, _RAG_ROOT)
from eval_adapter import retrieve_chunks

# -------------------------------------------------------------------
# Judge configuration: OpenAI (preferred) > Bedrock > skip
# -------------------------------------------------------------------
_judge_model = None

# Option 1: OpenAI (check for real API key first)
_OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if _OPENAI_KEY and _OPENAI_KEY.startswith("sk-"):
    from openai_judge import get_openai_judge
    _judge_model = get_openai_judge()
    print(f"[DeepEval] Using {_judge_model.get_model_name()}")
else:
    # Option 2: AWS Bedrock (fallback)
    _AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
    _AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
    if _AWS_KEY and _AWS_SECRET:
        try:
            from bedrock_judge import get_bedrock_judge
            _judge_model = get_bedrock_judge()
            print(f"[DeepEval] Using {_judge_model.get_model_name()}")
        except ImportError:
            pass

    if _judge_model is None:
        pytest.skip(
            "DeepEval judge not configured. Set a real OPENAI_API_KEY, "
            "or AWS credentials for Bedrock, or run 'deepeval set-ollama <model>'.",
            allow_module_level=True,
        )

# Metric focused on retrieval quality (chunk relevance)
# Lower threshold - chunks naturally contain extra context beyond just the answer
ctx_rel = ContextualRelevancyMetric(threshold=0.25, model=_judge_model, include_reason=True)

@pytest.mark.parametrize("query", [
    "Welche Module werden laut aktuellem Lehrplan im Studiengang Bachelor Informatik angeboten?",
    "was sind die unterschiede zwischen den Prüfungsordnungen inf14 und inf20",
    "Welche Module sind laut dem Modulhandbuch B_Inf14.0 im dritten Semester zu bestehen?",
    "Was ist die regelstudienzeit im Bachelor Informatik?",
    "Was sind die Unterschiede zwischen den Studiengängen informatik und Medieninformatik?",
    "Was sind die Vorraussetzungen für eine Zulassung zum Bachelor Studium an der FH-Wedel?",
    "Wie laufen Prüfungen grundsätzlich ab?",
])
def test_retrieval_only(query):
    chunks = retrieve_chunks(query)

    # Print verbose output for debugging (encode to handle special chars on Windows)
    def safe_print(text):
        try:
            print(text)
        except UnicodeEncodeError:
            print(text.encode('ascii', 'replace').decode('ascii'))
    
    safe_print(f"\n{'='*60}")
    safe_print(f"QUERY: {query}")
    safe_print(f"{'='*60}")
    safe_print(f"Retrieved {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        preview = chunk[:200].replace('\n', ' ') + "..." if len(chunk) > 200 else chunk.replace('\n', ' ')
        safe_print(f"  [{i}] {preview}")
    safe_print(f"{'='*60}\n")

    test_case = LLMTestCase(
        input=query,
        actual_output="(retrieval only)",
        retrieval_context=chunks,
    )

    assert_test(test_case, [ctx_rel])
