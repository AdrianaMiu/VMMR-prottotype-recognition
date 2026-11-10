"""Load persisted prototype tables into a Qdrant target (issue M5-1, PROTOCOL.md §12).

The M1 gate writes one table per track and variant at
``<prototypes_dir>/<encoder>/<variant>/m0_<protocol_id>.parquet`` with a ``.sha256`` sidecar
(:mod:`avr.prototypes.persist`). This module walks that layout, refuses a table whose sidecar
disagrees, derives the §12 collection name from the table itself (prototype method, encoder,
preprocessing id, dimension) plus the protocol id in the file name and the mapping version of
the track, and exports every table through :func:`avr.retrieval.qdrant_store.export_prototypes`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow.parquet as pq

from avr.io import read_sidecar_sha256
from avr.pipeline.gate import MAPPING_VERSION
from avr.retrieval.qdrant_store import (
    ExportSummary,
    QdrantTarget,
    collection_name,
    export_prototypes,
)

TABLE_PREFIX = "m0_"


def _single(table, column: str) -> str:
    values = set(table.column(column).to_pylist())
    if len(values) != 1:
        raise ValueError(f"{column} must be constant within a prototype table, got {values}")
    return str(values.pop())


def collection_for_table(path: Path, table) -> str:
    """The §12 collection name of the table stored at ``path``."""
    stem = path.stem
    if not stem.startswith(TABLE_PREFIX):
        raise ValueError(f"{path.name}: expected a '{TABLE_PREFIX}<protocol_id>' file name")
    protocol_id = stem[len(TABLE_PREFIX) :]
    track = next((t for t in MAPPING_VERSION if protocol_id.startswith(f"{t}_")), None)
    if track is None:
        raise ValueError(f"{path.name}: protocol id {protocol_id!r} names no known track")
    return collection_name(
        protocol_id,
        MAPPING_VERSION[track],
        _single(table, "encoder"),
        _single(table, "preprocessing_id"),
        int(_single(table, "dimension")),
        _single(table, "prototype_method"),
    )


def prototype_table_paths(prototypes_dir: Path) -> list[Path]:
    paths = sorted(Path(prototypes_dir).glob(f"*/*/{TABLE_PREFIX}*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no {TABLE_PREFIX}*.parquet tables under {prototypes_dir}")
    return paths


def load_prototype_tables(prototypes_dir: Path, target: QdrantTarget) -> list[ExportSummary]:
    """Export every checked prototype table under ``prototypes_dir`` to ``target``."""
    summaries = []
    for path in prototype_table_paths(prototypes_dir):
        sidecar = path.with_suffix(".sha256")
        if not sidecar.exists():
            raise ValueError(f"{path}: no .sha256 sidecar")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != read_sidecar_sha256(sidecar):
            raise ValueError(f"{path}: sha256 {digest} differs from the sidecar {sidecar.name}")
        table = pq.read_table(path)
        summaries.append(export_prototypes(target, collection_for_table(path, table), table))
    return summaries
