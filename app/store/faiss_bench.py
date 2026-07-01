import time
from typing import Dict

import numpy as np
import faiss


def faiss_benchmark(num_vectors: int = 100000, dim: int = 768, k: int = 10) -> Dict[str, float]:
    xb = np.random.random((num_vectors, dim)).astype("float32")
    xq = np.random.random((1000, dim)).astype("float32")
    index = faiss.IndexFlatL2(dim)
    start = time.time()
    index.add(xb)
    insert_time = time.time() - start
    start = time.time()
    index.search(xq, k)
    query_time = time.time() - start
    return {"num_vectors": float(num_vectors), "dim": float(dim), "insert_time_s": insert_time, "query_time_s": query_time, "qps": 1000 / query_time}
