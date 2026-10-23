import pytest
from PIL import Image

from wildsearch.images import load_image


def test_converts_to_rgb(tmp_path):
    path1 = tmp_path / "gray.png"
    path2 = tmp_path / "transparent.png"
    Image.new("L", (10, 10), 128).save(path1)
    Image.new("RGBA", (10, 10), (255, 0, 0, 0)).save(path2)
    assert load_image(path1).mode == "RGB"
    assert load_image(path2).mode == "RGB"


def test_applies_exif_rotation(tmp_path):
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90° CW
    Image.new("RGB", (40, 20)).save(path, exif=exif)  # stored landscape
    assert load_image(path).size == (20, 40)  # displayed portrait


def test_broken_file_raises_oserror(tmp_path):
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"this is not an image")
    with pytest.raises(OSError):
        load_image(path)
