"""Shard-resumable embedding extraction over the image manifest (PROTOCOL.md §9, ADR-0002).

Jobs are the valid manifest rows in ``image_id`` order joined with the private path table
(:func:`avr.preprocessing.p3_batch.load_jobs`). For P3, :func:`select_p3_jobs` keeps only the
images whose mask metadata says ``ok`` and whose mask PNG exists; every other image is
recorded as :class:`MissingP3` and excluded, never replaced by its P1 crop. Jobs are then
split into fixed-size shards; a shard whose Parquet and manifest already exist in the
:class:`~avr.encoders.store.EmbeddingStore` is skipped, so a rerun resumes at shard granularity.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from pydantic import BaseModel

from avr.encoders.store import EmbeddingStore
from avr.preprocessing.p3 import load_mask, mask_path
from avr.preprocessing.p3_batch import P3Job, load_jobs
from avr.preprocessing.variants import Variant, prepare

ImageJob = P3Job
MISSING_NAME = "missing.json"


class Embedder(Protocol):
    native_dim: int

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray: ...


class MissingP3(BaseModel):
    """An image excluded from P3: ``reason`` is the mask status or a file-level cause."""

    image_id: str
    reason: str


@dataclass
class ExtractionStats:
    images_embedded: int = 0
    shards_written: list[int] = field(default_factory=list)
    shards_skipped: list[int] = field(default_factory=list)
    elapsed_s: float = 0.0


def select_p3_jobs(
    jobs: Sequence[ImageJob], mask_metadata: Path, masks_dir: Path
) -> tuple[list[ImageJob], list[MissingP3]]:
    """Jobs with an ``ok`` mask on disk, and the excluded ones with their reason."""
    table = pq.read_table(mask_metadata, columns=["image_id", "mask_status"]).to_pydict()
    status = dict(zip(table["image_id"], table["mask_status"], strict=True))
    kept, missing = [], []
    for job in jobs:
        mask_status = status.get(job.image_id)
        if mask_status is None:
            missing.append(MissingP3(image_id=job.image_id, reason="no_mask_metadata"))
        elif mask_status != "ok":
            missing.append(MissingP3(image_id=job.image_id, reason=mask_status))
        elif not mask_path(masks_dir, job.image_id).exists():
            missing.append(MissingP3(image_id=job.image_id, reason="mask_file_missing"))
        else:
            kept.append(job)
    return kept, missing


def write_missing(store: EmbeddingStore, missing: Sequence[MissingP3]) -> Path:
    store.directory.mkdir(parents=True, exist_ok=True)
    out = store.directory / MISSING_NAME
    payload = {
        "variant": store.identity.variant,
        "count": len(missing),
        "missing": [m.model_dump() for m in missing],
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out


def _prepare_batch(
    root: Path, jobs: Sequence[ImageJob], variant: Variant, masks_dir: Path | None, expand: float
) -> list[Image.Image]:
    crops = []
    for job in jobs:
        with Image.open(Path(root) / job.relative_path) as image:
            image = image.convert("RGB")
        mask = None
        if variant == "p3":
            if masks_dir is None:
                raise ValueError("variant 'p3' needs masks_dir")
            mask = load_mask(mask_path(masks_dir, job.image_id))
        crops.append(prepare(image, job.bbox_xyxy, variant, mask=mask, expand=expand))
    return crops


def run_extraction(
    root: Path,
    jobs: Sequence[ImageJob],
    encoder: Embedder,
    store: EmbeddingStore,
    variant: Variant,
    batch_size: int,
    shard_size: int,
    masks_dir: Path | None = None,
    *,
    expand: float,
    on_shard: Callable[[int, int, ExtractionStats], None] | None = None,
) -> ExtractionStats:
    """Embed ``jobs`` shard by shard into ``store``; finished shards are not recomputed."""
    stats = ExtractionStats()
    started = time.perf_counter()
    n_shards = (len(jobs) + shard_size - 1) // shard_size
    for index in range(n_shards):
        if store.has_shard(index):
            stats.shards_skipped.append(index)
            continue
        chunk = jobs[index * shard_size : (index + 1) * shard_size]
        blocks = []
        for start in range(0, len(chunk), batch_size):
            batch = chunk[start : start + batch_size]
            crops = _prepare_batch(root, batch, variant, masks_dir, expand)
            blocks.append(encoder.embed(crops))
        store.write_shard(index, [j.image_id for j in chunk], np.vstack(blocks))
        stats.images_embedded += len(chunk)
        stats.shards_written.append(index)
        stats.elapsed_s = time.perf_counter() - started
        if on_shard is not None:
            on_shard(index, n_shards, stats)
    stats.elapsed_s = time.perf_counter() - started
    return stats


__all__ = [
    "ExtractionStats",
    "ImageJob",
    "MissingP3",
    "load_jobs",
    "run_extraction",
    "select_p3_jobs",
    "write_missing",
]
