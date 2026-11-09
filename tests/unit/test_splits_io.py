"""Shared split-manifest writer (ADR-0002: public manifests carry ids and hashes only)."""

import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from avr.data.splits_io import read_sidecar_sha256, write_split_manifest


def test_write_split_manifest_writes_parquet_and_a_sha256sum_style_sidecar(tmp_path: Path):
    table = pa.table({"image_id": ["b", "a"], "split_track_a": ["query", "support"]})

    out = write_split_manifest(table, tmp_path / "splits" / "official.parquet")

    assert out.manifest == tmp_path / "splits" / "official.parquet"
    assert out.sha256 == tmp_path / "splits" / "official.sha256"
    digest = hashlib.sha256(out.manifest.read_bytes()).hexdigest()
    assert out.sha256.read_text() == f"{digest}  official.parquet\n"
    assert read_sidecar_sha256(out.sha256) == digest
    assert pq.read_table(out.manifest).equals(table)


def test_write_split_manifest_is_byte_identical_across_calls(tmp_path: Path):
    table = pa.table({"image_id": ["a"], "n": [1]})

    first = write_split_manifest(table, tmp_path / "one.parquet")
    second = write_split_manifest(table, tmp_path / "two.parquet")

    assert first.manifest.read_bytes() == second.manifest.read_bytes()
