from pathlib import Path
from typing import Any, Dict, List, Optional

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
        if lancedb is not None:
            self.client = lancedb.connect(str(self.path))
            tables = self.client.list_tables()
            self.table = self.client.table(self.table_name) if self.table_name in tables else None

    def create_table(self) -> Any:
        if lancedb is None:
            raise RuntimeError("lancedb is not installed")
        if self.table is None:
            self.table = self.client.create_table(self.table_name)
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
            self.table.upsert(records)
            logger.info("Upserted {} documents", len(records))

    def similarity_search(self, vector: List[float], k: int = 5, filter: Optional[Dict[str, Any]] = None):
        if self.table is None:
            return []
        q = self.table.search(vector).limit(k)
        if filter:
            for key, value in filter.items():
                q = q.filter(f"{key} == '{value}'")
        return q.execute()

    def metadata_filter_search(self, filter: Dict[str, Any], k: int = 10):
        if self.table is None:
            return []
        q = self.table.search().limit(k)
        for key, value in filter.items():
            q = q.filter(f"{key} == '{value}'")
        return q.execute()

    def delete_document(self, doc_id: str) -> bool:
        if self.table is None:
            return False
        self.table.delete_where(f"id == '{doc_id}'")
        return True

    def list_documents(self, limit: int = 100) -> List[str]:
        if self.table is None:
            return []
        return list(self.table.to_arrow_table().to_pydict().get("id", []))[:limit]
