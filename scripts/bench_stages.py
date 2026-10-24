"""Times each stage of a search separately, plus peak memory.

Usage: uv run python scripts/bench_stages.py ~/datasets/cub-bench "a red bird"
"""

import resource
import sys
import time
from pathlib import Path

from wildsearch.clip import ClipEmbedder
from wildsearch.images import load_image
from wildsearch.scanner import scan
from wildsearch.search import rank

folder, query = Path(sys.argv[1]), sys.argv[2]
timings: dict[str, float] = {}


def stage(name: str, start: float) -> float:
    """Record the time since `start` under `name`; return a new start time."""
    now = time.perf_counter()
    timings[name] = now - start
    return now


t = time.perf_counter()
embedder = ClipEmbedder()
t = stage("load model", t)

paths = [f.path for f in scan(folder)]
t = stage("scan", t)

images = [load_image(p) for p in paths]
t = stage("decode", t)

image_vecs = embedder.embed_images(images)
t = stage("embed images", t)

query_vec = embedder.embed_texts([query])[0]
t = stage("embed query", t)

rank(image_vecs, query_vec, n=10)
t = stage("rank", t)

total = sum(timings.values())
for name, seconds in timings.items():
    print(f"{name:<14}{seconds:8.2f} s  {seconds / total:6.1%}")
print(f"{'total':<14}{total:8.2f} s")

# ru_maxrss is in bytes on macOS (but in kilobytes on Linux).
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(f"peak memory   {peak / 1e9:8.2f} GB  ({len(paths)} photos)")
