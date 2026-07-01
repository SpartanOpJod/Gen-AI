from app.ingest.ingestor import chunk_text, sha256_text


def test_sha256_text():
    assert len(sha256_text("hello")) == 64


def test_chunk_text():
    text = " ".join([f"word{i}" for i in range(20)])
    chunks = chunk_text(text, chunk_size=5, overlap=1)
    assert len(chunks) > 0
