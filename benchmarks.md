# Benchmarks

How long a search takes and how much memory it needs, measured stage by stage.
Each optimization gets its own section below, measured on the same two folders.

## Setup

- **Machine:** Mac M2 Max, 64 GB RAM. CPU only.
- **Model:** OpenCLIP ViT-B/32 with LAION-2B weights (`ViT-B-32:laion2b_s34b_b79k`).
- **Query:** `"a red bird"`, top 10.
- **Scripts:** `scripts/bench_info.py` describes a benchmarking folder; `scripts/bench_stages.py` times
  each stage and reports peak memory (`ru_maxrss`).

Two benchmarking folders on purpose, at opposite ends:

| Folder | Photos | Size | Megapixels (median, range) | Notes |
|---|---|---|---|---|
| **CUB** (many, small pictures) | 11,788 | 1.15 GB | 0.18 (0.02–0.25) | Full image set of [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011). |
| **Large** (a few larger pictures)| 23 | 0.61 GB | 39.8 (7.7–39.8) | My own camera JPEGs. Not public; represents real camera output. |

## Baseline: in-memory, unbatched ( first version of the project )

Every photo is decoded and held in memory, then embedded in a single call.


### CUB (11,788 small photos)

| Stage        | Time         | Share  |
|--------------|--------------|--------|
| load model   | 1.77 s       | 1.5 %  |
| scan         | 0.17 s       | 0.1 %  |
| decode       | 16.11 s      | 13.4 % |
| embed images | 101.72 s     | 84.9 % |
| embed query  | 0.05 s       | 0.0 %  |
| rank         | 0.00 s       | 0.0 %  |
| **total**    | **119.81 s** |        |

Peak memory: **41.90 GB**.

### Large (23 camera photos)

| Stage        | Time        | Share  |
|--------------|-------------|--------|
| load model   | 1.56 s      | 14.5 % |
| scan         | 0.00 s      | 0.0 %  |
| decode       | 5.88 s      | 54.5 % |
| embed images | 3.34 s      | 31.0 % |
| embed query  | 0.02 s      | 0.2 %  |
| rank         | 0.00 s      | 0.0 %  |
| **total**    | **10.79 s** |        |

Peak memory: **4.85 GB**.

### What the baseline shows

- **Many small photos: the model dominates** (85 % on CUB, about 8.6 ms per photo).
- **Few large photos: pixel handling dominates.** Decoding is 55 %, and "embed images"
  costs ~145 ms per photo instead of ~8.6 ms, because it includes resizing each 40 MP
  image to 224 px before the model sees it.
- **The query side is nearly free:** embedding the query and ranking 11,788 vectors take
  about 0.05 s. Storing the photo vectors once (indexing) should therefore make repeated
  searches take seconds instead of minutes.
- **Memory grows with the whole folder:** 41.9 GB for CUB, 4.85 GB for just 23 large
  photos.

**Conclusion:** every search re-processes the entire folder, and that is what has to go.
- Photos should be embedded once and stored (indexing), so a search only embeds the query
- Images should be processed in batches, so memory no longer grows with the folder; and
- Large photos should be decoded at reduced size, since most of their pixels are discarded before the model sees them.
