"""Describe a benchmark folder: number of photos, total size, typical resolution.

Usage: uv run python scripts/bench_info.py ~/datasets/cub-bench
"""

import statistics
import sys
from pathlib import Path

from PIL import Image

from wildsearch.scanner import scan

files = list(scan(Path(sys.argv[1])))
megapixels = []
for f in files:
    with Image.open(f.path) as img:  # reads the header only, no decoding
        megapixels.append(img.width * img.height / 1e6)

print(f"photos:      {len(files)}")
print(f"total size:  {sum(f.size for f in files) / 1e9:.2f} GB")
print(
    f"megapixels:  median {statistics.median(megapixels):.2f}, "
    f"min {min(megapixels):.2f}, max {max(megapixels):.2f}"
)
