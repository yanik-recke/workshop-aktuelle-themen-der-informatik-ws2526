import os
import sys
import os.path as op
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

# Require a judge model API key for DeepEval (OpenAI recommended)
if not os.getenv("OPENAI_API_KEY"):
    pytest.skip(
        "Set OPENAI_API_KEY (or configure a judge via deepeval CLI) to run DeepEval metrics.",
        allow_module_level=True,
    )

# Configure metrics (tune thresholds as you collect results)
answer_rel = AnswerRelevancyMetric(threshold=0.6, model="gpt-4o-mini", include_reason=True)
faithful = FaithfulnessMetric(threshold=0.7, model="gpt-4o-mini", include_reason=True)
ctx_rel = ContextualRelevancyMetric(threshold=0.6, model="gpt-4o-mini", include_reason=True)

@pytest.mark.parametrize("query", [
    "Welche Module werden laut aktuellem Lehrplan im Studiengang Bachelor Informatik angeboten?",
    "was sind die unterschiede zwischen den Prüfungsordnungen inf14 und inf20",
    "Welche Module sind laut dem Modulhandbuch B_Inf14.0 im dritten Semester zu bestehen?",
    "Was ist die regelstudienzeit im Bachelor Informatik?",
    "Was sind die Unterschiede zwischen den Studiengängen informatik und Medieninformatik?",
    "Was sind die Vorraussetzungen für eine Zulassung zum Bachelor Studium an der FH-Wedel?",
    "Wie laufen Prüfungen grundsätzlich ab?",
])
def test_rag_quality(query):
    answer, retrieval_context = run_single_turn(query)

    test_case = LLMTestCase(
        input=query,
        actual_output=answer,
        retrieval_context=retrieval_context,  # list[str] of retrieved chunks
    )

    assert_test(test_case, [answer_rel, faithful, ctx_rel])
