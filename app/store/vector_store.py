from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa

from app.core.config import settings
from app.core.logging import logger

try:
    import lancedb
except Exception:
    lancedb = None


class VectorStoreManager:
    def __init__(self, table_name: str = "documents"):
        self.table_name = table_name
        self.path = Path(settings.lancedb_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = None
        self.table = None
        self._documents: List[Dict[str, Any]] = []
        if lancedb is not None:
            self.client = lancedb.connect(str(self.path))
            try:
                tables = self.client.list_tables()
            except AttributeError:
                tables = getattr(self.client, "table_names", lambda: [])()
            if self.table_name in tables:
                try:
                    self.table = self.client.table(self.table_name)
                except AttributeError:
                    self.table = self.client.open_table(self.table_name)

    def create_table(self) -> Any:
        if lancedb is None:
            raise RuntimeError("lancedb is not installed")
        if self.table is None:
            schema = pa.schema([
                ("id", pa.string()),
                ("source", pa.string()),
                ("document_type", pa.string()),
                ("page_number", pa.int64()),
                ("chunk_id", pa.string()),
                ("doc_hash", pa.string()),
                ("ingestion_timestamp", pa.string()),
                ("text", pa.string()),
                ("embedding", pa.list_(pa.float32())),
                ("embedding_model", pa.string()),
                ("embedding_dim", pa.int64()),
            ])
            self.table = self.client.create_table(self.table_name, schema=schema)
        return self.table

    def upsert_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        if self.table is None:
            self.create_table()
        records = []
        try:
            existing_hashes = set(self.table.to_arrow_table().to_pydict().get("doc_hash", []))
        except Exception:
            existing_hashes = set()

        for doc, emb in zip(documents, embeddings):
            meta = doc["metadata"]
            if meta.get("doc_hash") in existing_hashes:
                continue
            records.append({**meta, "text": doc["text"], "embedding": emb, "embedding_model": settings.embedding_model, "embedding_dim": settings.embedding_dim})

        if records:
            if self.table is not None:
                self.table.add(records)
            else:
                self._documents.extend([
                    {
                        **record,
                        "embedding_model": settings.embedding_model,
                        "embedding_dim": settings.embedding_dim,
                    }
                    for record in records
                ])
            logger.info("Upserted {} documents", len(records))

    def similarity_search(self, vector: List[float], k: int = 5, filter: Optional[Dict[str, Any]] = None):
        if self.table is not None:
            try:
                q = self.table.search(vector, vector_column_name="embedding").limit(k)
                if filter:
                    for key, value in filter.items():
                        q = q.where(f"{key} == '{value}'")
                return q.to_list()
            except Exception as exc:
                logger.warning("Falling back to in-memory search because LanceDB query failed: {}", exc)

        docs = self._documents
        if filter:
            docs = [doc for doc in docs if all(doc.get(key) == value for key, value in filter.items())]

        if not docs:
            return []

        def cosine_similarity(a: List[float], b: List[float]) -> float:
            if not a or not b:
                return 0.0
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(y * y for y in b) ** 0.5
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        scored = []
        for doc in docs:
            emb = doc.get("embedding") or []
            scored.append((cosine_similarity(vector, emb), doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:k]]

    def metadata_filter_search(self, filter: Dict[str, Any], k: int = 10):
        if self.table is None:
            return []
        q = self.table.search().limit(k)
        for key, value in filter.items():
            q = q.where(f"{key} == '{value}'")
        return q.to_list()

    def delete_document(self, doc_id: str) -> bool:
        if self.table is None:
            return False
        self.table.delete_where(f"id == '{doc_id}'")
        return True

    def list_documents(self, limit: int = 100) -> List[str]:
        if self.table is None:
            return []
        return list(self.table.to_arrow_table().to_pydict().get("id", []))[:limit]
