"""Interrupt-and-resume of scripts/extract_embeddings.py on 50-image shards (issue M1-13).

Needs COMPCARS_ROOT, the built manifest and path table, and a CUDA device. The "interrupt"
is a ``limit`` that stops after the first shard; the rerun with a larger limit must leave
shard 0 untouched (same mtime, same bytes) and only embed the second shard.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

pytestmark = pytest.mark.skipif(
    not (
        os.environ.get("COMPCARS_ROOT")
        and torch.cuda.is_available()
        and (REPO_ROOT / "data" / "paths" / "compcars_web_image_paths.parquet").exists()
    ),
    reason="needs COMPCARS_ROOT, the private path table and a CUDA device",
)


def test_second_run_resumes_after_the_finished_shard(tmp_path: Path):
    import extract_embeddings

    base = [
        "encoders@encoder=dinov3_vitb16",
        "extraction.variant=p1",
        "extraction.shard_size=50",
        "extraction.batch_size=16",
        f"extraction.embeddings_dir={tmp_path / 'emb'}",
        f"tracking.tracking_dir={tmp_path / 'mlruns'}",
    ]
    first = extract_embeddings.main([*base, "extraction.limit=50"])
    assert first.images_embedded == 50 and first.shards_written == [0]
    shard0 = tmp_path / "emb" / "dinov3_vitb16" / "p1" / "shard_00000.parquet"
    mtime = shard0.stat().st_mtime_ns
    digest = hashlib.sha256(shard0.read_bytes()).hexdigest()

    second = extract_embeddings.main([*base, "extraction.limit=100"])

    assert second.shards_skipped == [0]
    assert second.shards_written == [1]
    assert second.images_embedded == 50
    assert shard0.stat().st_mtime_ns == mtime
    assert hashlib.sha256(shard0.read_bytes()).hexdigest() == digest
    assert (shard0.parent / "shard_00001.parquet").exists()
