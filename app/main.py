from typing import Any
import time

from fastapi import FastAPI, HTTPException

from app.core.config import settings
from app.core.logging import logger
from app.embeddings.encoder import EmbeddingModel
from app.ingest.ingestor import ingest_path
from app.llm.llm import DummyLLM
from app.models import AskRequest, AskResponse, IngestRequest
from app.retrieval.rerank import rerank_by_cosine
from app.store.vector_store import VectorStoreManager

app = FastAPI(title="RAG Service")

vstore = VectorStoreManager()
llm = DummyLLM()
embedder = EmbeddingModel()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict[str, Any]:
    docs = ingest_path(req.path, req.chunk_size, req.overlap)
    if not docs:
        raise HTTPException(status_code=400, detail="No supported documents found")
    texts = [doc["text"] for doc in docs]
    embeddings = embedder.embed_texts(texts)
    vstore.upsert_documents(docs, embeddings)
    return {"ingested": len(docs)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    start = time.time()
    query_vector = embedder.embed_texts([req.question])[0]
    hits = vstore.similarity_search(query_vector, k=req.k, filter=req.filter)
    contexts = []
    for hit in hits:
        if isinstance(hit, dict):
            contexts.append(hit)
    reranked = rerank_by_cosine(query_vector, contexts) if contexts else contexts
    answer = llm.generate(req.question, reranked)
    latency_ms = int((time.time() - start) * 1000)
    return AskResponse(answer=answer.get("answer", ""), citations=answer.get("citations", []), latency_ms=latency_ms)


@app.get("/documents")
def list_documents() -> dict[str, Any]:
    return {"documents": vstore.list_documents()}
