from app.evaluation.judge import JudgeRequest, simple_reference_judge
from app.llm.llm import DummyLLM
from app.evaluation.judge import llm_as_judge


def test_simple_judge():
    req = JudgeRequest(input="q", system_prompt="", model_output="hello world", expected_output="hello world", criteria=["correctness"])
    result = simple_reference_judge(req)
    assert result["overall_score"] >= 1


def test_llm_judge_fallback():
    req = JudgeRequest(input="q", system_prompt="", model_output="hello", expected_output="hello", criteria=["correctness"])
    result = llm_as_judge(req, DummyLLM())
    assert isinstance(result, dict)
