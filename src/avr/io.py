"""File digests, ``.sha256`` sidecars, fixed-size-list columns and repo-relative paths.

Every published table (image manifest, mask metadata, split tables, prototype tables,
embedding shards; ADR-0002) is paired with a sidecar in ``sha256sum`` format:
``<hex digest>  <file name>``. Every report path written under ``reports/`` or
``data/**/*.json`` is relative to the repository root so no machine-specific prefix is
ever committed. This module is the single home of those conventions.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pyarrow as pa

REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_of_file(path: Path, chunk_size: int = 1 << 22) -> str:
    """Hex SHA-256 of the bytes at ``path``, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sha256_sidecar(path: Path) -> Path:
    """Write ``<digest>  <name>`` to ``path.with_suffix('.sha256')`` and return that path."""
    path = Path(path)
    sidecar = path.with_suffix(".sha256")
    sidecar.write_text(f"{sha256_of_file(path)}  {path.name}\n")
    return sidecar


def read_sidecar_sha256(sidecar: Path) -> str:
    """The hex digest recorded in a ``sha256sum``-style sidecar."""
    return Path(sidecar).read_text().split()[0]


def read_fixed_list_matrix(table: pa.Table, column: str) -> np.ndarray:
    """A fixed-size-list column of ``table`` as a ``[rows, list_size]`` float32 matrix."""
    chunked = table.column(column)
    if not pa.types.is_fixed_size_list(chunked.type):
        raise TypeError(f"column {column!r} is {chunked.type}, not a fixed-size list")
    combined = chunked.combine_chunks()
    values = np.asarray(combined.values.to_numpy(zero_copy_only=False), dtype=np.float32)
    return values.reshape(table.num_rows, combined.type.list_size)


def repo_relative(path: Path, root: Path = REPO_ROOT) -> str:
    """``path`` relative to ``root`` (``../`` steps when outside), never absolute."""
    return os.path.relpath(Path(path).resolve(), Path(root).resolve())
