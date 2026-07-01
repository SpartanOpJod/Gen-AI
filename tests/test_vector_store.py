from app.store.vector_store import VectorStoreManager


def test_similarity_search_uses_in_memory_records_when_lancedb_is_unavailable():
    store = VectorStoreManager(table_name="test_fallback")
    store._documents = [
        {"chunk_id": "alpha", "embedding": [1.0, 0.0]},
        {"chunk_id": "beta", "embedding": [0.0, 1.0]},
    ]

    hits = store.similarity_search([1.0, 0.0], k=1)

    assert len(hits) == 1
    assert hits[0]["chunk_id"] == "alpha"
