from pathlib import Path

from PIL import Image, ImageOps


def load_image(path: Path) -> Image.Image:
    """
    Returns the image, converted to RGB image, required by CLIP
    """
    with Image.open(path) as img:
        upright = ImageOps.exif_transpose(img)
        return upright.convert("RGB")
