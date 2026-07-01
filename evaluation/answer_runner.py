import json
from pathlib import Path
from typing import Any, Dict, List

from app.evaluation.judge import JudgeRequest, simple_reference_judge
from app.llm.llm import DummyLLM


def run_answer_evaluation(questions_path: str, out_path: str = "evaluation/answer_results.json") -> List[Dict[str, Any]]:
    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    llm = DummyLLM()
    output = []
    for q in questions:
        gen = llm.generate(q["question"], [])
        req = JudgeRequest(input=q["question"], system_prompt="", model_output=gen.get("answer", ""), expected_output=q.get("gold_answer", ""), criteria=["correctness", "faithfulness", "completeness"])
        judged = simple_reference_judge(req)
        output.append({"question": q["question"], "answer": gen.get("answer", ""), "judge": judged})
    Path(out_path).write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output
