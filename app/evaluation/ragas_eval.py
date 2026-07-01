from typing import Any, Dict


def evaluate_with_ragas(answer: str, context: str, reference: str) -> Dict[str, Any]:
    ans_tokens = set(answer.lower().split())
    ref_tokens = set(reference.lower().split())
    ctx_tokens = set(context.lower().split())
    return {
        "faithfulness": round(len(ans_tokens & ref_tokens) / max(1, len(ref_tokens)), 3),
        "answer_relevancy": round(len(ans_tokens & ref_tokens) / max(1, len(ans_tokens)), 3),
        "context_precision": round(len(ans_tokens & ctx_tokens) / max(1, len(ans_tokens)), 3),
        "context_recall": round(len(ref_tokens & ctx_tokens) / max(1, len(ref_tokens)), 3),
    }
