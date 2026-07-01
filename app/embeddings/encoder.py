import hashlib
from typing import List, Optional

from app.core.config import settings
from app.core.logging import logger

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


class EmbeddingModel:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.embedding_model
        if SentenceTransformer is None:
            logger.warning("sentence-transformers not available; using deterministic fallback embeddings")
            self.model = None
            self.dim = settings.embedding_dim
        else:
            self.model = SentenceTransformer(self.model_name)
            self.dim = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self.model is None:
            result = []
            for text in texts:
                digest = hashlib.sha256(text.lower().encode("utf-8")).digest()
                vector = []
                for byte in digest:
                    vector.append(byte / 255.0)
                while len(vector) < settings.embedding_dim:
                    vector.extend(vector[: min(8, settings.embedding_dim - len(vector))])
                result.append(vector[: settings.embedding_dim])
            return result
        embs = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [emb.tolist() for emb in embs]

    def info(self) -> dict:
        return {"model": self.model_name, "dim": self.dim}
