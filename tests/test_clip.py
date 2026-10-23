"""Tests for the real CLIP embedder.

These need the `ml` extra (torch + open_clip) and download the model weights on the
first run. CI doesn't install the extra, so the whole file is skipped there.
Run locally with: uv run pytest tests/test_clip.py
"""

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("open_clip")  # no open_clip installed → skip this whole file

from wildsearch.clip import ClipEmbedder

MODEL = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


@pytest.fixture(scope="module")  # load the model once for all tests in this file
def clip() -> ClipEmbedder:
    return ClipEmbedder(MODEL, PRETRAINED)


def solid(color: str) -> Image.Image:
    return Image.new("RGB", (300, 200), color)


def test_model_id(clip: ClipEmbedder) -> None:
    assert clip.model_id == f"{MODEL}:{PRETRAINED}"


def test_image_vectors_follow_the_contract(clip: ClipEmbedder) -> None:
    vecs = clip.embed_images([solid("red"), solid("blue"), solid("green")])
    assert isinstance(vecs, np.ndarray)
    assert vecs.ndim == 2 and vecs.shape[0] == 3
    assert vecs.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1, rtol=1e-6)


def test_text_vectors_follow_the_contract(clip: ClipEmbedder) -> None:
    vecs = clip.embed_texts(["a lion", "a car", "a plate of pasta"])
    assert isinstance(vecs, np.ndarray)
    assert vecs.ndim == 2 and vecs.shape[0] == 3
    assert vecs.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1, rtol=1e-6)


def test_images_and_texts_share_a_space(clip: ClipEmbedder) -> None:
    img = clip.embed_images([solid("red")])
    txt = clip.embed_texts(["red"])
    assert img.shape[1] == txt.shape[1]


def test_same_input_gives_same_vector(clip: ClipEmbedder) -> None:
    # model.eval() makes inference deterministic; without it, dropout could vary.
    a = clip.embed_images([solid("green")])
    b = clip.embed_images([solid("green")])
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_images_match_their_description(clip: ClipEmbedder) -> None:
    images = clip.embed_images([solid("red"), solid("blue")])
    texts = clip.embed_texts(["a red image", "a blue image"])
    scores = images @ texts.T  # scores[i, j] = similarity of image i and text j
    assert scores[0, 0] > scores[0, 1]  # red image prefers "red"
    assert scores[1, 1] > scores[1, 0]  # blue image prefers "blue"
