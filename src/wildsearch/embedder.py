"""The interface every embedder follows."""

from typing import Protocol

import numpy as np
from PIL import Image


class Embedder(Protocol):
    model_id: str

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        """Return an array of shape (len(images), dim), dtype float32.

        Row i is the vector for images[i], scaled to length 1.
        """
        ...

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim), dtype float32.

        Row i is the vector for texts[i], scaled to length 1.
        """
        ...
