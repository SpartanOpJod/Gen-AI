import json
import time
from pathlib import Path
from statistics import median
from typing import Callable, List

from app.embeddings.encoder import EmbeddingModel
from app.llm.llm import DummyLLM
from app.store.vector_store import VectorStoreManager


def latency_benchmark(model_generate_fn: Callable[[str], dict], questions: List[dict], runs: int = 10, out_path: str = "evaluation/latency_results.json") -> dict:
    retrieval_latencies = []
    end_to_end_latencies = []
    embedder = EmbeddingModel()
    vstore = VectorStoreManager()
    for _ in range(runs):
        for q in questions:
            start = time.perf_counter()
            qvec = embedder.embed_texts([q["question"]])[0]
            vstore.similarity_search(qvec, k=5)
            retrieval_latencies.append((time.perf_counter() - start) * 1000)
            start_e2e = time.perf_counter()
            model_generate_fn(q["question"])
            end_to_end_latencies.append((time.perf_counter() - start_e2e) * 1000)
    retrieval_sorted = sorted(retrieval_latencies)
    e2e_sorted = sorted(end_to_end_latencies)
    result = {
        "retrieval_p50_ms": round(median(retrieval_sorted), 2),
        "retrieval_p95_ms": round(retrieval_sorted[int(len(retrieval_sorted) * 0.95) - 1] if len(retrieval_sorted) >= 20 else retrieval_sorted[-1], 2),
        "end_to_end_p95_ms": round(e2e_sorted[int(len(e2e_sorted) * 0.95) - 1] if len(e2e_sorted) >= 20 else e2e_sorted[-1], 2),
        "samples": len(retrieval_sorted),
    }
    Path(out_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
