from typing import Any, Dict, List
import numpy as np


def cosine_similarity(a: List[float], b: List[float]) -> float:
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    if np.linalg.norm(a_arr) == 0 or np.linalg.norm(b_arr) == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


def rerank_by_cosine(query_vector: List[float], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    scored = []
    for cand in candidates:
        emb = cand.get("embedding")
        score = cosine_similarity(query_vector, emb) if emb is not None else 0.0
        scored.append((score, cand))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [cand for _, cand in scored]
