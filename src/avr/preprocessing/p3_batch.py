"""Resumable P3 mask build over the image manifest (PROTOCOL.md §8, ADR-0002).

Jobs come from the public manifest (valid rows only, ``image_id`` order) joined with the
private path table. Work is split into fixed-size shards; each finished shard is one
Parquet file under ``shards_dir`` and a rerun skips every shard that already exists, so an
interrupted run resumes at shard granularity. ``merge_shards`` concatenates the shards into
``data/manifests/p3_mask_metadata.parquet`` with its ``.sha256`` and a summary JSON (same
publication discipline as the image manifest).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from pydantic import BaseModel

from avr.io import write_sha256_sidecar
from avr.preprocessing.p3 import DEFAULT_MIN_IOU, build_p3, mask_path, save_mask
from avr.segmentation.base import Segmenter

METADATA_NAME = "p3_mask_metadata"
METADATA_SCHEMA = pa.schema(
    [
        ("image_id", pa.string()),
        ("mask_status", pa.string()),
        ("gt_iou", pa.float32()),
        ("mask_area_fraction", pa.float32()),
        ("detection_score", pa.float32()),
    ]
)
METADATA_COLUMNS = tuple(METADATA_SCHEMA.names)


class P3Job(BaseModel):
    image_id: str
    relative_path: str
    bbox_xyxy: tuple[int, int, int, int]


class MetadataOutputs(BaseModel):
    metadata: Path
    sha256: Path
    summary: Path


def load_jobs(manifest_path: Path, paths_path: Path, limit: int | None = None) -> list[P3Job]:
    """Valid manifest rows in ``image_id`` order, joined with their private relative path."""
    public = pq.read_table(
        manifest_path, columns=["image_id", "bbox_xyxy", "validity_status"]
    ).to_pylist()
    private = pq.read_table(paths_path).to_pydict()
    paths = dict(zip(private["image_id"], private["relative_path"], strict=True))
    jobs = [
        P3Job(image_id=r["image_id"], relative_path=paths[r["image_id"]], bbox_xyxy=r["bbox_xyxy"])
        for r in sorted(public, key=lambda r: r["image_id"])
        if r["validity_status"] == "valid"
    ]
    return jobs if limit is None else jobs[:limit]


def shard_path(shards_dir: Path, index: int) -> Path:
    return Path(shards_dir) / f"shard_{index:05d}.parquet"


def run_shard(
    root: Path,
    jobs: list[P3Job],
    segmenter: Segmenter,
    masks_dir: Path,
    batch_size: int,
    expand: float = 0.05,
    min_iou: float = DEFAULT_MIN_IOU,
) -> list[dict]:
    rows = []
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start : start + batch_size]
        images = [Image.open(Path(root) / j.relative_path).convert("RGB") for j in chunk]
        for job, image, instances in zip(
            chunk, images, segmenter.segment_batch(images), strict=True
        ):
            result = build_p3(image, job.bbox_xyxy, instances, expand=expand, min_iou=min_iou)
            save_mask(result.mask, mask_path(masks_dir, job.image_id))
            rows.append(
                {
                    "image_id": job.image_id,
                    "mask_status": result.status,
                    "gt_iou": result.gt_iou,
                    "mask_area_fraction": result.mask_area_fraction,
                    "detection_score": result.detection_score,
                }
            )
    return rows


def run_batch(
    root: Path,
    jobs: list[P3Job],
    segmenter: Segmenter,
    masks_dir: Path,
    shards_dir: Path,
    shard_size: int,
    batch_size: int,
    expand: float = 0.05,
    min_iou: float = DEFAULT_MIN_IOU,
    on_shard: Callable[[int, int, int], None] | None = None,
) -> list[Path]:
    """Process ``jobs`` shard by shard; shards whose Parquet already exists are skipped."""
    shards_dir = Path(shards_dir)
    shards_dir.mkdir(parents=True, exist_ok=True)
    written = []
    n_shards = (len(jobs) + shard_size - 1) // shard_size
    for index in range(n_shards):
        path = shard_path(shards_dir, index)
        if path.exists():
            written.append(path)
            continue
        chunk = jobs[index * shard_size : (index + 1) * shard_size]
        rows = run_shard(root, chunk, segmenter, masks_dir, batch_size, expand, min_iou)
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pylist(rows, schema=METADATA_SCHEMA), tmp, compression="zstd")
        tmp.replace(path)
        written.append(path)
        if on_shard is not None:
            on_shard(index, n_shards, len(rows))
    return written


def summarise(rows: list[dict]) -> dict:
    statuses = Counter(r["mask_status"] for r in rows)
    ok = [r for r in rows if r["mask_status"] == "ok"]
    detected = [r for r in rows if r["detection_score"] is not None]

    def mean(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    return {
        "row_count": len(rows),
        "mask_status_counts": dict(sorted(statuses.items())),
        "mask_failure_rate": round(1 - len(ok) / len(rows), 4) if rows else None,
        "gt_iou_mean_ok": mean([r["gt_iou"] for r in ok]),
        "mask_area_fraction_mean_ok": mean([r["mask_area_fraction"] for r in ok]),
        "detection_score_mean": mean([r["detection_score"] for r in detected]),
    }


def merge_shards(shards_dir: Path, manifests_dir: Path) -> MetadataOutputs:
    shards = sorted(Path(shards_dir).glob("shard_*.parquet"))
    table = pa.concat_tables([pq.read_table(p, schema=METADATA_SCHEMA) for p in shards])
    table = table.sort_by("image_id")
    manifests_dir = Path(manifests_dir)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    out = MetadataOutputs(
        metadata=manifests_dir / f"{METADATA_NAME}.parquet",
        sha256=manifests_dir / f"{METADATA_NAME}.sha256",
        summary=manifests_dir / f"{METADATA_NAME}.summary.json",
    )
    pq.write_table(table, out.metadata, compression="zstd")
    write_sha256_sidecar(out.metadata)
    out.summary.write_text(json.dumps(summarise(table.to_pylist()), indent=2) + "\n")
    return out
