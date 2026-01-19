"""
RAG Quality Tests using DeepEval metrics.

Prerequisites:
    pip install pytest deepeval openai boto3

API Keys (configure in rag-code/.env):
    Option A - OpenAI (recommended):
        OPENAI_API_KEY="sk-proj-..."   # Must start with "sk-"
    
    Option B - AWS Bedrock (fallback):
        AWS_ACCESS_KEY_ID=ASIA...
        AWS_SECRET_ACCESS_KEY=...
        AWS_SESSION_TOKEN=...          # Expires every ~12h
        AWS_DEFAULT_REGION=eu-north-1

Running Tests:
    # Windows: use 'python -m' prefix (pytest/deepeval may not be in PATH)
    cd rag-code
    
    python -m pytest tests/test_rag_quality.py -v              # Run all tests
    python -m pytest tests/test_rag_quality.py -v -k "Informatik"  # Filter by keyword
    python -m deepeval test run tests/test_rag_quality.py      # With DeepEval dashboard

Metrics:
    - AnswerRelevancyMetric (threshold 0.5): Is the answer relevant to the question?
    - FaithfulnessMetric (threshold 0.5): Is the answer grounded in retrieved chunks?
    - ContextualRelevancyMetric (threshold 0.25): Are retrieved chunks relevant?
"""
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
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
)

# Make the rag-code folder importable for eval_adapter
# tests/ is inside rag-code/, so parent of this file is the rag-code root
_RAG_ROOT = op.dirname(op.dirname(__file__))
if _RAG_ROOT not in sys.path:
    sys.path.insert(0, _RAG_ROOT)
from eval_adapter import run_single_turn

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
        except ImportError as e:
            print(f"[DeepEval] Bedrock import failed: {e}")

    if _judge_model is None:
        pytest.skip(
            "DeepEval judge not configured. Set a real OPENAI_API_KEY, "
            "or AWS credentials for Bedrock, or run 'deepeval set-ollama <model>'.",
            allow_module_level=True,
        )

# -------------------------------------------------------------------
# Configure metrics with the selected judge
# -------------------------------------------------------------------
# Thresholds tuned for real-world RAG performance
answer_rel = AnswerRelevancyMetric(threshold=0.5, model=_judge_model, include_reason=True)
faithful = FaithfulnessMetric(threshold=0.5, model=_judge_model, include_reason=True)
ctx_rel = ContextualRelevancyMetric(threshold=0.25, model=_judge_model, include_reason=True)  # Chunks often contain extra context

normalQuery = ["Welche Module werden laut aktuellem Lehrplan im Studiengang Bachelor Informatik angeboten?",
              "Was ist die regelstudienzeit im Bachelor Informatik?",
              "Welche Module sind im dritten Semester des bachelors Informatik zu bestehen?",
              "Was sind die Vorraussetzungen für eine Zulassung zum Bachelor Studium an der FH-Wedel?",
              "Wie laufen Prüfungen grundsätzlich ab?",
              "Was ist bei einem Auslandssemester zu beachten?"
]

comparisonQuery = [
    "was sind die unterschiede zwischen den Prüfungsordnungen inf14 und inf20",
    "Was sind die Unterschiede zwischen den Studiengängen informatik und Medieninformatik?",
]


@pytest.mark.parametrize("query", [
    "Welche Module werden laut aktuellem Lehrplan im Studiengang Bachelor Informatik angeboten?",
    "was sind die unterschiede zwischen den Prüfungsordnungen inf14 und inf20",
    "Welche Module sind laut dem Modulhandbuch B_Inf14.0 im dritten Semester zu bestehen?",
    "Was ist die regelstudienzeit im Bachelor Informatik?",
    "Was sind die Unterschiede zwischen den Studiengängen informatik und Medieninformatik?",
    "Was sind die Vorraussetzungen für eine Zulassung zum Bachelor Studium an der FH-Wedel?",
    "Wie laufen Prüfungen grundsätzlich ab?",
    "Was ist bei einem Auslandssemester zu beachten?",
])
def test_rag_quality(query):
    answer, retrieval_context = run_single_turn(query)

    # Print verbose output for debugging (encode to handle special chars on Windows)
    def safe_print(text):
        try:
            print(text)
        except UnicodeEncodeError:
            print(text.encode('ascii', 'replace').decode('ascii'))
    
    safe_print(f"\n{'='*60}")
    safe_print(f"QUERY: {query}")
    safe_print(f"{'='*60}")
    safe_print(f"ANSWER: {answer[:300]}..." if len(answer) > 300 else f"ANSWER: {answer}")
    safe_print(f"\nRetrieved {len(retrieval_context)} chunks:")
    for i, chunk in enumerate(retrieval_context, 1):
        preview = chunk[:150].replace('\n', ' ') + "..." if len(chunk) > 150 else chunk.replace('\n', ' ')
        safe_print(f"  [{i}] {preview}")
    safe_print(f"{'='*60}\n")

    test_case = LLMTestCase(
        input=query,
        actual_output=answer,
        retrieval_context=retrieval_context,  # list[str] of retrieved chunks
    )

    assert_test(test_case, [answer_rel, faithful, ctx_rel])
