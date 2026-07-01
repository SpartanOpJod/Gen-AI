import json
from math import log2
from pathlib import Path
from typing import Any, Dict, List

from app.embeddings.encoder import EmbeddingModel
from app.ingest.ingestor import ingest_path
from app.store.vector_store import VectorStoreManager


def _infer_relevant_chunks(question: str) -> List[str]:
    lowered = question.lower()
    if "lancedb" in lowered or "pinecone" in lowered or "vector" in lowered:
        return ["vector_db-0"]
    if "judge" in lowered or "bias" in lowered or "sycophancy" in lowered:
        return ["judge_bias-0"]
    return ["rag_basics-0"]


def _dcg(hits: List[str], relevant: List[str], k: int) -> float:
    dcg = 0.0
    for rank, hit in enumerate(hits[:k], start=1):
        if hit in relevant:
            dcg += 1 / log2(rank + 1)
    return dcg


def _precision_at_k(hits: List[str], relevant: List[str], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant_hits = sum(1 for hit in hits[:k] if hit in relevant)
    return relevant_hits / k


def run_retrieval_evaluation(questions_path: str, k: int = 5, out_path: str = "evaluation/retrieval_results.json") -> Dict[str, Any]:
    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    embedder = EmbeddingModel()
    vstore = VectorStoreManager()
    sample_dir = Path("data/samples")
    if sample_dir.exists():
        docs = ingest_path(str(sample_dir))
        if docs:
            embeddings = embedder.embed_texts([doc["text"] for doc in docs])
            vstore.upsert_documents(docs, embeddings)

    per_query = []
    recall_scores = []
    hit_scores = []
    mrr_scores = []
    ndcg_scores = []
    context_precision_scores = []

    for q in questions:
        qvec = embedder.embed_texts([q["question"]])[0]
        hits = vstore.similarity_search(qvec, k=k)
        hit_ids = [h.get("chunk_id") if isinstance(h, dict) else str(h) for h in hits]
        relevant = q.get("relevant_chunks") or _infer_relevant_chunks(q["question"])
        relevant_set = set(relevant)
        relevant_hits = [hit for hit in hit_ids if hit in relevant_set]
        recall = len(relevant_hits) / len(relevant) if relevant else 0.0
        hit_rate = 1.0 if relevant_hits else 0.0
        mrr = 0.0
        for rank, hit in enumerate(hit_ids, start=1):
            if hit in relevant_set:
                mrr = 1.0 / rank
                break
        ideal_hits = [hit for hit in hit_ids if hit in relevant_set][:len(relevant)]
        idcg = _dcg(ideal_hits, relevant_set, k)
        dcg = _dcg(hit_ids, relevant_set, k)
        ndcg = dcg / idcg if idcg else 0.0
        cp = _precision_at_k(hit_ids, relevant_set, k)

        recall_scores.append(recall)
        hit_scores.append(hit_rate)
        mrr_scores.append(mrr)
        ndcg_scores.append(ndcg)
        context_precision_scores.append(cp)
        per_query.append({"question": q["question"], "hits": hit_ids, "relevant_chunks": relevant, "recall_at_k": round(recall, 3), "hit_rate": round(hit_rate, 3), "mrr": round(mrr, 3), "ndcg_at_k": round(ndcg, 3), "context_precision": round(cp, 3)})

    summary = {
        "recall_at_k": round(sum(recall_scores) / len(recall_scores), 3) if recall_scores else 0.0,
        "hit_rate": round(sum(hit_scores) / len(hit_scores), 3) if hit_scores else 0.0,
        "mrr": round(sum(mrr_scores) / len(mrr_scores), 3) if mrr_scores else 0.0,
        "ndcg_at_k": round(sum(ndcg_scores) / len(ndcg_scores), 3) if ndcg_scores else 0.0,
        "context_precision": round(sum(context_precision_scores) / len(context_precision_scores), 3) if context_precision_scores else 0.0,
    }
    result = {"summary": summary, "per_query": per_query}
    Path(out_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return {"saved_to": out_path, "summary": summary}
