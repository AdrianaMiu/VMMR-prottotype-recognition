"""Build P3 vehicle masks for every valid manifest image (PROTOCOL.md §8, issue M1-19).

Runs the frozen RF-DETR segmenter from configs/segmentation over the public manifest joined
with the private path table, writes one PNG mask per image under data/masks/ (git-ignored)
and one Parquet shard per ``--shard-size`` images under data/masks/_shards/; rerunning skips
finished shards. The shards are merged into data/manifests/p3_mask_metadata.{parquet,sha256,
summary.json}.
The low-overlap threshold (``min_iou``) and the P1 expansion come from the configs, never from
the command line.
Usage: uv run python scripts/build_p3_masks.py [--limit N] [--shard-size 1000] [--batch-size 16]
"""

from __future__ import annotations

import argparse
import json
import time

from avr.config import CONFIG_DIR, load_config
from avr.preprocessing.p3_batch import load_jobs, merge_shards, run_batch
from avr.segmentation.rfdetr_seg import RFDETRSegmenter

REPO_ROOT = CONFIG_DIR.parent
MANIFESTS_DIR = REPO_ROOT / "data" / "manifests"
PATHS_DIR = REPO_ROOT / "data" / "paths"
MASKS_DIR = REPO_ROOT / "data" / "masks"
SHARDS_DIR = MASKS_DIR / "_shards"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="first N valid images only")
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=None, help="default: config value")
    args = parser.parse_args()

    cfg = load_config()
    root = (REPO_ROOT / cfg.dataset.root).resolve()
    jobs = load_jobs(
        MANIFESTS_DIR / "compcars_web_images.parquet",
        PATHS_DIR / "compcars_web_image_paths.parquet",
        limit=args.limit,
    )
    total_valid = len(
        load_jobs(
            MANIFESTS_DIR / "compcars_web_images.parquet",
            PATHS_DIR / "compcars_web_image_paths.parquet",
        )
    )
    segmenter = RFDETRSegmenter.from_config(cfg.segmentation)
    print(f"checkpoint {segmenter.checkpoint_path} sha256 {segmenter.checkpoint_sha256}")

    processed = 0

    def on_shard(index: int, n_shards: int, n_rows: int) -> None:
        nonlocal processed
        processed += n_rows
        rate = processed / (time.perf_counter() - started)
        print(f"shard {index + 1}/{n_shards}: {processed} images, {rate:.1f} img/s", flush=True)

    started = time.perf_counter()
    run_batch(
        root,
        jobs,
        segmenter,
        MASKS_DIR,
        SHARDS_DIR,
        shard_size=args.shard_size,
        batch_size=args.batch_size or int(cfg.segmentation.batch_size),
        expand=float(cfg.preprocessing.bbox_expand_fraction),
        min_iou=float(cfg.segmentation.min_iou),
        on_shard=on_shard,
    )
    elapsed = time.perf_counter() - started
    out = merge_shards(SHARDS_DIR, MANIFESTS_DIR)
    summary = json.loads(out.summary.read_text())
    print(out.summary.read_text())
    if processed:
        rate = processed / elapsed
        print(
            f"segmented {processed} new images in {elapsed:.0f}s ({rate:.2f} img/s); "
            f"full valid manifest ({total_valid}) extrapolates to {total_valid / rate / 60:.0f} min"
        )
    print(f"mask failure rate: {summary['mask_failure_rate']}")


if __name__ == "__main__":
    main()
