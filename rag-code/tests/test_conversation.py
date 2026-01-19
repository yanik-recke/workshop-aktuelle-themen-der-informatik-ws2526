"""
Test conversation quality using DeepEval's ConversationalTestCase.
Tests multi-turn conversation completeness and coherence.
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
from deepeval.test_case import LLMTestCase, ConversationalTestCase
from deepeval.metrics import ConversationCompletenessMetric

# Make the rag-code folder importable for eval_adapter
_RAG_ROOT = op.dirname(op.dirname(__file__))
if _RAG_ROOT not in sys.path:
    sys.path.insert(0, _RAG_ROOT)
from eval_adapter import run_single_turn

# -------------------------------------------------------------------
# Judge configuration: OpenAI (preferred) > Bedrock > skip
# -------------------------------------------------------------------
_judge_model = None

_OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if _OPENAI_KEY and _OPENAI_KEY.startswith("sk-"):
    from openai_judge import get_openai_judge
    _judge_model = get_openai_judge()
    print(f"[DeepEval] Using {_judge_model.get_model_name()}")
else:
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
            "or AWS credentials for Bedrock.",
            allow_module_level=True,
        )

# Conversation completeness metric
conv_completeness = ConversationCompletenessMetric(
    threshold=0.6,
    model=_judge_model,
    include_reason=True,
)


def run_conversation(turns: list[str]) -> list[LLMTestCase]:
    """Run a multi-turn conversation and return test cases for each turn."""
    test_cases = []
    for query in turns:
        answer, retrieval_context = run_single_turn(query)
        
        # Print verbose output
        print(f"\n  USER: {query}")
        print(f"  BOT: {answer[:200]}..." if len(answer) > 200 else f"  BOT: {answer}")
        
        test_cases.append(LLMTestCase(
            input=query,
            actual_output=answer,
            retrieval_context=retrieval_context,
        ))
    return test_cases


# Define multi-turn conversation scenarios
CONVERSATION_SCENARIOS = [
    {
        "name": "Bachelor Informatik inquiry",
        "turns": [
            "Was ist die Regelstudienzeit im Bachelor Informatik?",
            "Welche Module sind im ersten Semester?",
            "Gibt es Praktika in diesem Studiengang?",
        ],
    },
    {
        "name": "Study program comparison",
        "turns": [
            "Welche Bachelorstudiengänge gibt es an der FH Wedel?",
            "Was sind die Unterschiede zwischen Informatik und Medieninformatik?",
            "Welcher Studiengang ist für Spieleentwicklung besser geeignet?",
        ],
    },
    {
        "name": "Admission requirements",
        "turns": [
            "Was sind die Zulassungsvoraussetzungen für ein Bachelorstudium?",
            "Brauche ich ein Praktikum vor dem Studium?",
            "Kann ich mich auch ohne Abitur bewerben?",
        ],
    },
]


@pytest.mark.parametrize(
    "scenario",
    CONVERSATION_SCENARIOS,
    ids=[s["name"] for s in CONVERSATION_SCENARIOS],
)
def test_conversation_completeness(scenario):
    """Test that multi-turn conversations are complete and coherent."""
    print(f"\n{'='*60}")
    print(f"CONVERSATION: {scenario['name']}")
    print(f"{'='*60}")
    
    test_cases = run_conversation(scenario["turns"])
    
    print(f"{'='*60}\n")
    
    convo_test_case = ConversationalTestCase(
        turns=test_cases,
    )
    
    assert_test(convo_test_case, [conv_completeness])
