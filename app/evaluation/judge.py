import json
from datetime import datetime
from pathlib import Path
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


def simple_reference_judge(req: JudgeRequest) -> Dict[str, Any]:
    overlap = len(set(req.model_output.split()) & set(req.expected_output.split()))
    total = max(1, len(set(req.expected_output.split())))
    score = max(1, min(5, round(5 * (overlap / total))))
    out = {c: {"score": score, "rationale": "Simple lexical overlap heuristic."} for c in req.criteria}
    out["overall_score"] = round(sum(v["score"] for v in out.values()) / max(1, len(out)), 2)
    entry = {"timestamp": datetime.utcnow().isoformat(), "result": out}
    with open(JUDGE_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return out


def llm_as_judge(req: JudgeRequest, llm) -> Dict[str, Any]:
    system = "You are an automated judge. Return JSON with each criterion score 1-5 and rationale, plus overall_score."
    prompt = f"Input: {req.input}\nModel Output: {req.model_output}\nExpected Output: {req.expected_output}\nCriteria: {req.criteria}"
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
