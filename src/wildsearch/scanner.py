import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ScannedFile:
    path: Path  # absolute, resolved
    size: int  # bytes
    mtime_ns: int  # modification time


def scan(directory: Path) -> Iterator[ScannedFile]:
    """Yield every image under `directory`, recursively, with absolute paths."""
    root = directory.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {directory}")
    yield from _walk(root)


def is_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS


def _walk(directory: Path) -> Iterator[ScannedFile]:
    try:
        entries = list(os.scandir(directory))
    except OSError:
        return  # unreadable folder: skip it instead of stopping the whole scan

    for entry in entries:
        if entry.name.startswith("."):
            continue
        if entry.is_dir(follow_symlinks=False):  # avoids symlink loops
            yield from _walk(Path(entry.path))
        elif is_image(entry.name):
            path = Path(entry.path)
            try:
                stats = path.stat()
            except OSError:
                continue  # file vanished or is unreadable
            yield ScannedFile(path, stats.st_size, stats.st_mtime_ns)
