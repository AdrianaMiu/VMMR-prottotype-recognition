"""Shared writer for public split manifests (PROTOCOL.md §5; ADR-0002).

Both tracks publish their split tables under ``data/splits/<track>/`` as Parquet plus a
``.sha256`` sidecar in ``sha256sum`` format. The table is written with a fixed schema
and ``zstd`` compression so a rebuild is byte-identical; callers sort rows before
calling. Only opaque ``image_id`` and ``sha256`` values identify images; no path is
ever written here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel


class SplitManifestOutputs(BaseModel):
    manifest: Path
    sha256: Path


def write_split_manifest(table: pa.Table, path: Path) -> SplitManifestOutputs:
    """Write ``table`` to ``path`` and its SHA-256 to ``path.with_suffix('.sha256')``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = path.with_suffix(".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n")
    return SplitManifestOutputs(manifest=path, sha256=sidecar)


def read_sidecar_sha256(sidecar: Path) -> str:
    """The hex digest recorded in a ``sha256sum``-style sidecar."""
    return sidecar.read_text().split()[0]
