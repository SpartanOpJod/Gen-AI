import json
from pathlib import Path
from typing import Any, Dict, List

from app.evaluation.judge import JudgeRequest, simple_reference_judge
from app.evaluation.ragas_eval import evaluate_with_ragas
from app.embeddings.encoder import EmbeddingModel
from app.llm.llm import DummyLLM
from app.store.vector_store import VectorStoreManager


def run_answer_evaluation(questions_path: str, out_path: str = "evaluation/answer_results.json") -> List[Dict[str, Any]]:
    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    llm = DummyLLM()
    embedder = EmbeddingModel()
    vstore = VectorStoreManager()
    output = []
    for q in questions:
        qvec = embedder.embed_texts([q["question"]])[0]
        hits = vstore.similarity_search(qvec, k=5)
        contexts = [h.get("text") if isinstance(h, dict) else str(h) for h in hits]
        gen = llm.generate(q["question"], contexts)
        req = JudgeRequest(input=q["question"], system_prompt="", model_output=gen.get("answer", ""), expected_output=q.get("gold_answer", ""), criteria=["correctness", "faithfulness", "completeness"])
        judged = simple_reference_judge(req)
        ragas = evaluate_with_ragas(gen.get("answer", ""), "\n".join(contexts), q.get("gold_answer", ""))
        output.append({"question": q["question"], "answer": gen.get("answer", ""), "judge": judged, "ragas": ragas, "citations": gen.get("citations", []), "token_usage": gen.get("token_usage", {})})
    Path(out_path).write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output
