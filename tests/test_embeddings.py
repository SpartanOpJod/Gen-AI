from app.embeddings.encoder import EmbeddingModel


def test_embedding_fallback_uses_nonzero_vectors():
    model = EmbeddingModel()
    embeddings = model.embed_texts(["retrieval augmented generation", "vector database"])

    assert len(embeddings) == 2
    assert all(len(embedding) > 0 for embedding in embeddings)
    assert any(abs(value) > 1e-9 for embedding in embeddings for value in embedding)
