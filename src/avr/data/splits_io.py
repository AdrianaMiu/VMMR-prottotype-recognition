"""Shared writer for public split manifests (PROTOCOL.md §5; ADR-0002).

Both tracks publish their split tables under ``data/splits/<track>/`` as Parquet plus a
``.sha256`` sidecar in ``sha256sum`` format. The table is written with a fixed schema
and ``zstd`` compression so a rebuild is byte-identical; callers sort rows before
calling. Only opaque ``image_id`` and ``sha256`` values identify images; no path is
ever written here.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from avr.io import write_sha256_sidecar


class SplitManifestOutputs(BaseModel):
    manifest: Path
    sha256: Path


def write_split_manifest(table: pa.Table, path: Path) -> SplitManifestOutputs:
    """Write ``table`` to ``path`` and its SHA-256 to ``path.with_suffix('.sha256')``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    return SplitManifestOutputs(manifest=path, sha256=write_sha256_sidecar(path))
