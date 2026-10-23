"""CLIP-style models via open_clip. The only module that imports PyTorch."""

import numpy as np
import open_clip
import torch
from PIL import Image


def _to_unit_numpy(features: torch.Tensor) -> np.ndarray:
    """Scale each row to length 1 and return it as a float32 numpy array."""
    features = features / features.norm(dim=-1, keepdim=True)
    return features.float().numpy()


class ClipEmbedder:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        self.model_id = f"{model_name}:{pretrained}"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.eval()
        self.tokeniser = open_clip.get_tokenizer(model_name)

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        preprocessed_images = torch.stack([self.preprocess(img) for img in images])
        with torch.no_grad():
            image_features = self.model.encode_image(preprocessed_images)
        return _to_unit_numpy(image_features)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        preprocessed_texts = self.tokeniser(texts)
        with torch.no_grad():
            text_features = self.model.encode_text(preprocessed_texts)
        return _to_unit_numpy(text_features)
