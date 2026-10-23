from pathlib import Path

import numpy as np

from wildsearch.embedder import Embedder
from wildsearch.images import load_image
from wildsearch.scanner import scan


def rank(image_vecs: np.ndarray, query_vec: np.ndarray, n: int) -> list[tuple[int, float]]:
    """Return the n best matches as (row index, score), best first.

    Both inputs must be normalized, so the dot product is cosine similarity.
    Returns fewer than n results if there are fewer photos.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    scores = image_vecs @ query_vec
    ordered_indices = np.argsort(-scores)[:n]
    return [(int(idx), float(scores[idx])) for idx in ordered_indices]


def search_folder(embedder: Embedder, folder: Path, query: str, n: int) -> list[tuple[Path, float]]:
    """Return the n photos in `folder` that best match `query`, best first."""
    paths = [f.path for f in scan(folder)]
    if not paths:
        return []
    images = [load_image(p) for p in paths]
    image_vecs = embedder.embed_images(images)
    query_vec = embedder.embed_texts([query])[0]
    matches = rank(image_vecs, query_vec, n)
    return [(paths[i], score) for i, score in matches]
