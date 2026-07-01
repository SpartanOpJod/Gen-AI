import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup

try:
    import fitz
except Exception:
    fitz = None

from app.core.config import settings
from app.core.logging import logger


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    tokens = text.split()
    chunks: List[str] = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


def extract_text_from_pdf(path: Path) -> List[Dict[str, Any]]:
    if fitz is None:
        raise RuntimeError("PyMuPDF not installed")
    doc = fitz.open(path)
    return [{"page_number": i + 1, "text": page.get_text()} for i, page in enumerate(doc)]


def extract_text_from_html(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(text, "html.parser")
    body = soup.get_text(separator="\n")
    return [{"page_number": 1, "text": body}]


def extract_text_from_md(path: Path) -> List[Dict[str, Any]]:
    return [{"page_number": 1, "text": Path(path).read_text(encoding="utf-8")}]


def ingest_path(path: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None) -> List[Dict[str, Any]]:
    p = Path(path)
    cs = chunk_size or settings.default_chunk_size
    ov = overlap or settings.default_chunk_overlap
    docs: List[Dict[str, Any]] = []
    files = list(p.rglob("*")) if p.is_dir() else [p]

    for f in files:
        if not f.is_file():
            continue
        suffix = f.suffix.lower()
        try:
            if suffix == ".pdf":
                pages = extract_text_from_pdf(f)
                doc_type = "pdf"
            elif suffix in {".html", ".htm"}:
                pages = extract_text_from_html(f)
                doc_type = "html"
            elif suffix in {".md", ".markdown"}:
                pages = extract_text_from_md(f)
                doc_type = "md"
            else:
                continue

            for page in pages:
                text = page.get("text", "")
                doc_hash = sha256_text(text)
                for idx, chunk in enumerate(chunk_text(text, cs, ov)):
                    docs.append({
                        "text": chunk,
                        "metadata": {
                            "id": f"{f.stem}-{page.get('page_number')}-{idx}",
                            "source": str(f.resolve()),
                            "document_type": doc_type,
                            "page_number": page.get("page_number"),
                            "chunk_id": f"{f.stem}-{page.get('page_number')}-{idx}",
                            "doc_hash": doc_hash,
                            "ingestion_timestamp": datetime.utcnow().isoformat(),
                        },
                    })
        except Exception as exc:
            logger.exception("Failed to ingest {}: {}", f, exc)

    return docs
