import json
from pathlib import Path
from typing import Any, Dict

from app.embeddings.encoder import EmbeddingModel
from app.store.vector_store import VectorStoreManager


def run_retrieval_evaluation(questions_path: str, k: int = 5, out_path: str = "evaluation/retrieval_results.json") -> Dict[str, Any]:
    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    embedder = EmbeddingModel()
    vstore = VectorStoreManager()
    results = []
    for q in questions:
        qvec = embedder.embed_texts([q["question"]])[0]
        hits = vstore.similarity_search(qvec, k=k)
        results.append({"question": q["question"], "hits": [h.get("chunk_id") if isinstance(h, dict) else str(h) for h in hits]})
    Path(out_path).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return {"saved_to": out_path, "count": len(results)}
