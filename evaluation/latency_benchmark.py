import json
import time
from pathlib import Path
from statistics import median
from typing import Callable, List


def latency_benchmark(model_generate_fn: Callable[[str], dict], questions: List[dict], runs: int = 10, out_path: str = "evaluation/latency_results.json") -> dict:
    latencies = []
    for _ in range(runs):
        for q in questions:
            start = time.time()
            model_generate_fn(q["question"])
            latencies.append((time.time() - start) * 1000)
    lat_sorted = sorted(latencies)
    result = {
        "p50_ms": round(median(lat_sorted), 2),
        "p95_ms": round(lat_sorted[int(len(lat_sorted) * 0.95) - 1] if len(lat_sorted) >= 20 else lat_sorted[-1], 2),
        "samples": len(lat_sorted),
    }
    Path(out_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
