"""Shard-resumable embedding extraction with a fake encoder (no GPU, no dataset)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from avr.encoders.extract import (
    MissingP3,
    run_extraction,
    select_p3_jobs,
    write_missing,
)
from avr.encoders.store import EmbeddingStore, StoreIdentity
from avr.preprocessing.p3 import save_mask
from avr.preprocessing.p3_batch import P3Job

DIM = 4


class FakeEncoder:
    """Deterministic 4-D unit vectors from the mean colour; counts images embedded."""

    native_dim = DIM

    def __init__(self) -> None:
        self.calls = 0
        self.embedded = 0
        self.corner_pixels: list[tuple[int, int, int]] = []

    def embed(self, images):
        self.calls += 1
        self.embedded += len(images)
        rows = []
        for image in images:
            self.corner_pixels.append(image.getpixel((0, 0)))
            mean = np.asarray(image, dtype=np.float32).reshape(-1, 3).mean(axis=0)
            vector = np.concatenate([mean, [1.0]]).astype(np.float32)
            rows.append(vector / np.linalg.norm(vector))
        return np.vstack(rows)


def _identity(variant: str) -> StoreIdentity:
    return StoreIdentity(
        encoder="fake",
        revision="rev",
        checkpoint_sha256="0" * 64,
        preprocessing_id="fake_v1",
        variant=variant,
        dtype="float32",
    )


def _jobs(root: Path, n: int) -> list[P3Job]:
    jobs = []
    for i in range(n):
        rel = f"image/{i}.jpg"
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40, 30), (10 * i + 5, 20, 30)).save(path, format="JPEG", quality=100)
        jobs.append(P3Job(image_id=f"{i:016x}", relative_path=rel, bbox_xyxy=(10, 5, 30, 25)))
    return jobs


def test_rerun_skips_finished_shards_and_only_embeds_the_new_ones(tmp_path: Path):
    root = tmp_path / "compcar"
    jobs = _jobs(root, 5)
    store = EmbeddingStore(tmp_path / "emb", _identity("p1"))
    encoder = FakeEncoder()

    first = run_extraction(
        root, jobs[:2], encoder, store, "p1", batch_size=8, shard_size=2, expand=0.05
    )
    assert first.images_embedded == 2 and first.shards_written == [0]
    assert encoder.embedded == 2
    mtime = store.shard_path(0).stat().st_mtime_ns

    second = run_extraction(
        root, jobs, encoder, store, "p1", batch_size=2, shard_size=2, expand=0.05
    )

    assert second.shards_skipped == [0]
    assert second.shards_written == [1, 2]
    assert second.images_embedded == 3
    assert encoder.embedded == 5, "shard 0 must not be recomputed"
    assert store.shard_path(0).stat().st_mtime_ns == mtime
    ids, matrix = store.load_all()
    assert ids == [j.image_id for j in jobs]
    assert matrix.shape == (5, DIM) and matrix.dtype == np.float32


def test_p1_crops_are_embedded_at_native_crop_size(tmp_path: Path):
    root = tmp_path / "compcar"
    jobs = _jobs(root, 1)
    encoder = FakeEncoder()
    store = EmbeddingStore(tmp_path / "emb", _identity("p1"))
    run_extraction(root, jobs, encoder, store, "p1", batch_size=1, shard_size=1, expand=0.05)
    # corner of the P1 crop is an image pixel (red channel 5 +- JPEG noise), not padding
    assert encoder.corner_pixels[0] != (127, 127, 127)


def test_p3_jobs_exclude_failed_or_missing_masks_and_embed_masked_crops(tmp_path: Path):
    root = tmp_path / "compcar"
    jobs = _jobs(root, 4)
    masks_dir = tmp_path / "masks"
    mask = np.zeros((30, 40), dtype=bool)
    mask[10:20, 15:25] = True
    for job in jobs[:3]:  # job 3 has metadata but no PNG
        save_mask(mask, masks_dir / f"{job.image_id}.png")
    metadata = tmp_path / "p3_mask_metadata.parquet"
    pq.write_table(
        pa.table(
            {
                "image_id": [j.image_id for j in jobs[:4]],
                "mask_status": ["ok", "no_detection", "ok", "ok"],
            }
        ),
        metadata,
    )

    kept, missing = select_p3_jobs(jobs + _jobs(root / "extra", 1)[:0], metadata, masks_dir)

    assert [j.image_id for j in kept] == [jobs[0].image_id, jobs[2].image_id]
    assert missing == [
        MissingP3(image_id=jobs[1].image_id, reason="no_detection"),
        MissingP3(image_id=jobs[3].image_id, reason="mask_file_missing"),
    ]

    encoder = FakeEncoder()
    store = EmbeddingStore(tmp_path / "emb", _identity("p3"))
    stats = run_extraction(
        root,
        kept,
        encoder,
        store,
        "p3",
        batch_size=8,
        shard_size=10,
        masks_dir=masks_dir,
        expand=0.05,
    )
    assert stats.images_embedded == 2
    assert encoder.corner_pixels == [(127, 127, 127), (127, 127, 127)]

    out = write_missing(store, missing)
    assert out == store.directory / "missing.json"
    assert '"mask_file_missing"' in out.read_text()


def test_images_without_mask_metadata_are_missing_too(tmp_path: Path):
    root = tmp_path / "compcar"
    jobs = _jobs(root, 2)
    metadata = tmp_path / "meta.parquet"
    pq.write_table(pa.table({"image_id": [jobs[0].image_id], "mask_status": ["ok"]}), metadata)
    save_mask(np.ones((30, 40), bool), tmp_path / "masks" / f"{jobs[0].image_id}.png")

    kept, missing = select_p3_jobs(jobs, metadata, tmp_path / "masks")

    assert [j.image_id for j in kept] == [jobs[0].image_id]
    assert missing == [MissingP3(image_id=jobs[1].image_id, reason="no_mask_metadata")]
