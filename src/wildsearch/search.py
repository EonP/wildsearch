from pathlib import Path

import numpy as np

from wildsearch.embedder import Embedder
from wildsearch.store import Store


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


def search_index(
    store: Store, embedder: Embedder, folder: Path, query: str, n: int
) -> list[tuple[Path, float]]:
    """Return the n indexed photos in `folder` that best match `query`, best first.

    Reads the stored vectors; no photo is opened. Empty if the folder isn't indexed.
    """
    paths, image_vecs = store.load_vectors(embedder.model_id, folder)
    if not paths:
        return []
    query_vec = embedder.embed_texts([query])[0]
    return [(paths[i], score) for i, score in rank(image_vecs, query_vec, n)]


# def get_image_vecs(embedder: Embedder, paths: list[Path], batch_size: int = 32) -> np.ndarray:
#     """Embeds photos in batches of `batch_size`. Row i belongs to paths[i]."""
#     if batch_size < 1:
#         raise ValueError(f"batch_size must be at least 1, got {batch_size}")
#     batches = []
#     for batch in batched(paths, batch_size):
#         images = [load_image(p) for p in batch]
#         batches.append(embedder.embed_images(images))
#     return np.vstack(batches)
