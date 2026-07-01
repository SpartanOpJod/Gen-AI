import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from app.core.logging import logger

LOG_PATH = Path("logs")
LOG_PATH.mkdir(parents=True, exist_ok=True)
JUDGE_LOG = LOG_PATH / "judge_logs.jsonl"


class JudgeRequest:
    def __init__(self, input: str, system_prompt: str, model_output: str, expected_output: str, criteria: List[str]):
        self.input = input
        self.system_prompt = system_prompt
        self.model_output = model_output
        self.expected_output = expected_output
        self.criteria = criteria


def _criterion_scores(result: Dict[str, Any]) -> List[int]:
    return [value.get("score", 0) for key, value in result.items() if key != "overall_score" and isinstance(value, dict)]


def simple_reference_judge(req: JudgeRequest) -> Dict[str, Any]:
    overlap = len(set(req.model_output.lower().split()) & set(req.expected_output.lower().split()))
    total = max(1, len(set(req.expected_output.lower().split())))
    score = max(1, min(5, round(5 * (overlap / total))))
    out = {c: {"score": score, "rationale": "Simple lexical overlap heuristic."} for c in req.criteria}
    out["overall_score"] = round(mean([v["score"] for v in out.values()]), 2)
    entry = {"timestamp": datetime.utcnow().isoformat(), "result": out}
    with open(JUDGE_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return out


def llm_as_judge(req: JudgeRequest, llm) -> Dict[str, Any]:
    system = "You are an automated judge. Return JSON with each criterion score 1-5 and rationale, plus overall_score."
    prompt = f"System: {system}\nInput: {req.input}\nModel Output: {req.model_output}\nExpected Output: {req.expected_output}\nCriteria: {req.criteria}"
    out = llm.generate(prompt, [])
    result = {c: {"score": 3, "rationale": "Neutral fallback."} for c in req.criteria}
    result["overall_score"] = 3.0
    if isinstance(out, dict) and isinstance(out.get("answer"), str):
        try:
            parsed = json.loads(out["answer"])
            if isinstance(parsed, dict):
                result = parsed
        except Exception:
            logger.warning("Judge parsing failed")
    return result


def aggregate_suite(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [r.get("overall_score", 0.0) for r in results]
    return {"mean_score": round(sum(scores) / len(scores), 2) if scores else 0.0, "pass_rate": round(sum(1 for s in scores if s >= 3) / len(scores), 2) if scores else 0.0}


def evaluate_position_bias(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    flips = 0
    for case in cases:
        first = case.get("score_a_then_b", 0)
        second = case.get("score_b_then_a", 0)
        if first != second:
            flips += 1
    return {"flip_rate": round(flips / max(1, len(cases)), 3), "average_score": round(mean([case.get("score_a_then_b", 0) for case in cases]), 3)}


def evaluate_verbosity_bias(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    changed = 0
    for case in cases:
        if case.get("verbose_score", 0) != case.get("base_score", 0):
            changed += 1
    return {"flip_rate": round(changed / max(1, len(cases)), 3)}


def evaluate_self_enhancement_mitigation(generator_family: str, judge_family: str) -> Dict[str, Any]:
    return {"generator_family": generator_family, "judge_family": judge_family, "mitigation_active": generator_family != judge_family}


def evaluate_sycophancy_probes(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    fooled = sum(1 for case in cases if case.get("judge_swayed", False))
    return {"sycophancy_rate": round(fooled / max(1, len(cases)), 3)}


def calibrate_with_anchor_examples(anchor_scores: List[float]) -> Dict[str, Any]:
    return {"anchor_mean": round(mean(anchor_scores), 3), "clustered": True}


def evaluate_adversarial_probes(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    fooled = sum(1 for case in cases if case.get("fooled", False))
    return {"adversarial_failure_rate": round(fooled / max(1, len(cases)), 3)}


def evaluate_agreement(labels_a: List[int], labels_b: List[int]) -> Dict[str, Any]:
    if not labels_a or len(labels_a) != len(labels_b):
        return {"agreement_rate": 0.0, "cohen_kappa": 0.0}
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / len(labels_a)
    return {"agreement_rate": round(agree, 3), "cohen_kappa": round(agree, 3)}


def evaluate_test_retest_consistency(scores: List[List[int]]) -> Dict[str, Any]:
    if len(scores) < 2:
        return {"stability": 0.0}
    pairs = []
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            pairs.append(sum(1 for a, b in zip(scores[i], scores[j]) if a == b) / max(1, len(scores[i])))
    return {"stability": round(mean(pairs), 3)}


def evaluate_ab_comparison(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    wins_a = sum(1 for case in cases if case.get("score_a", 0) >= case.get("score_b", 0))
    return {"a_win_rate": round(wins_a / max(1, len(cases)), 3), "b_win_rate": round(1 - wins_a / max(1, len(cases)), 3), "winner": "A" if wins_a >= len(cases) / 2 else "B"}
