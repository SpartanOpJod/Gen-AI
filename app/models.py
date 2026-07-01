from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


class IngestRequest(BaseModel):
    path: str
    chunk_size: Optional[int] = None
    overlap: Optional[int] = None


class AskRequest(BaseModel):
    question: str
    k: Optional[int] = 5
    filter: Optional[Dict[str, Any]] = None


class AskResponse(BaseModel):
    answer: str
    citations: List[str]
    latency_ms: int


class DocumentMetadata(BaseModel):
    id: str
    source: str
    document_type: str
    page_number: Optional[int]
    chunk_id: str
    doc_hash: str
    ingestion_timestamp: datetime
