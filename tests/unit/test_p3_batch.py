"""Resumable per-shard P3 mask build (scripts/build_p3_masks.py backs onto this module)."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from avr.data.manifest import build_manifest, write_manifest
from avr.preprocessing.p3 import load_mask
from avr.preprocessing.p3_batch import (
    METADATA_COLUMNS,
    P3Job,
    load_jobs,
    merge_shards,
    run_batch,
)
from avr.segmentation.base import Instance


class FakeSegmenter:
    """Full-image car instance for every image, except black images get no detection."""

    def __init__(self) -> None:
        self.seen: list[tuple[int, int]] = []

    def segment(self, image):
        return self.segment_batch([image])[0]

    def segment_batch(self, images):
        out = []
        for image in images:
            self.seen.append(image.size)
            if image.getpixel((0, 0)) == (0, 0, 0):
                out.append([])
                continue
            w, h = image.size
            mask = np.ones((h, w), dtype=bool)
            out.append([Instance((0.0, 0.0, float(w), float(h)), mask, 0.9, "car")])
        return out


def _tree(root: Path, n: int = 5) -> list[P3Job]:
    jobs = []
    for i in range(n):
        rel = f"image/1/1/2010/img{i}.jpg"
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        colour = (0, 0, 0) if i == 4 else (200, 30, 30)
        Image.new("RGB", (20 + i, 20), colour).save(path, format="JPEG", quality=100)
        jobs.append(P3Job(image_id=f"{i:016x}", relative_path=rel, bbox_xyxy=(0, 0, 20 + i, 20)))
    return jobs


def test_run_batch_writes_one_shard_per_chunk_and_resumes_only_missing_shards(tmp_path: Path):
    root, masks, shards = tmp_path / "root", tmp_path / "masks", tmp_path / "shards"
    jobs = _tree(root)
    seg = FakeSegmenter()

    written = run_batch(
        root, jobs, seg, masks, shards, shard_size=2, batch_size=3, expand=0.05, min_iou=0.5
    )

    assert [p.name for p in written] == [
        "shard_00000.parquet",
        "shard_00001.parquet",
        "shard_00002.parquet",
    ]
    assert len(seg.seen) == 5
    assert sorted(p.name for p in masks.iterdir()) == [f"{i:016x}.png" for i in range(5)]
    assert load_mask(masks / "0000000000000002.png").shape == (20, 22)

    (shards / "shard_00001.parquet").unlink()
    seg.seen.clear()
    run_batch(root, jobs, seg, masks, shards, shard_size=2, batch_size=3, expand=0.05, min_iou=0.5)

    assert seg.seen == [(22, 20), (23, 20)]


def test_merge_shards_builds_the_metadata_table_with_sha256_and_summary(tmp_path: Path):
    root, masks, shards, manifests = (
        tmp_path / n for n in ("root", "masks", "shards", "manifests")
    )
    jobs = _tree(root)
    run_batch(
        root,
        jobs,
        FakeSegmenter(),
        masks,
        shards,
        shard_size=2,
        batch_size=2,
        expand=0.05,
        min_iou=0.5,
    )

    out = merge_shards(shards, manifests)

    table = pq.read_table(out.metadata)
    assert table.column_names == list(METADATA_COLUMNS)
    assert table.column_names == [
        "image_id",
        "mask_status",
        "gt_iou",
        "mask_area_fraction",
        "detection_score",
    ]
    rows = table.to_pylist()
    assert [r["image_id"] for r in rows] == [f"{i:016x}" for i in range(5)]
    assert [r["mask_status"] for r in rows] == ["ok"] * 4 + ["no_detection"]
    assert rows[0]["gt_iou"] == 1.0 and rows[0]["mask_area_fraction"] == 1.0
    assert rows[4]["gt_iou"] is None and rows[4]["detection_score"] is None
    assert out.sha256.read_text() == (
        f"{hashlib.sha256(out.metadata.read_bytes()).hexdigest()}  p3_mask_metadata.parquet\n"
    )
    summary = json.loads(out.summary.read_text())
    assert summary["row_count"] == 5
    assert summary["mask_status_counts"] == {"no_detection": 1, "ok": 4}
    assert summary["mask_failure_rate"] == 0.2


def test_load_jobs_keeps_valid_manifest_rows_in_image_id_order(
    synthetic_compcars: Path, tmp_path: Path
):
    rows = build_manifest(synthetic_compcars, workers=1)
    out = write_manifest(rows, tmp_path / "manifests", tmp_path / "paths")

    jobs = load_jobs(out.manifest, out.paths)

    assert [j.relative_path for j in jobs] == ["image/1/1/2010/aaa.jpg"]
    assert jobs[0].bbox_xyxy == (1, 1, 6, 6)
    assert load_jobs(out.manifest, out.paths, limit=0) == []
