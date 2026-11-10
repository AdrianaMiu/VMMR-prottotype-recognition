"""Persisted prototype table (issue M1-16 addendum; PROTOCOL.md §12 payload fields).

One row per prototype: ``class_id``, the L2-normalised ``embedding``, ``support_count`` and
the keyword payload (make, model, year, generation, mapping status) plus the provenance that
makes the table citable (prototype method, encoder name/revision/checksum, preprocessing id,
dimension, run id). Rows are sorted by ``class_id`` and written with a fixed schema and
``zstd`` compression so a rebuild is byte-identical; the sidecar carries the SHA-256 of the
Parquet file in ``sha256sum`` format, like the split manifests (ADR-0002).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from avr.io import read_fixed_list_matrix, write_sha256_sidecar

PAYLOAD_COLUMNS = (
    "make_id",
    "make_name",
    "model_id",
    "model_name",
    "year",
    "generation_id",
    "mapping_status",
)


class PrototypeProvenance(BaseModel):
    model_config = ConfigDict(frozen=True)

    prototype_method: str
    encoder: str
    encoder_revision: str
    encoder_checksum: str
    preprocessing_id: str
    run_id: str


def prototype_schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            ("class_id", pa.string()),
            ("embedding", pa.list_(pa.float32(), dim)),
            ("support_count", pa.int32()),
            ("make_id", pa.int32()),
            ("make_name", pa.string()),
            ("model_id", pa.int32()),
            ("model_name", pa.string()),
            ("year", pa.int32()),
            ("generation_id", pa.string()),
            ("mapping_status", pa.string()),
            ("prototype_method", pa.string()),
            ("encoder", pa.string()),
            ("encoder_revision", pa.string()),
            ("encoder_checksum", pa.string()),
            ("preprocessing_id", pa.string()),
            ("dimension", pa.int32()),
            ("run_id", pa.string()),
        ]
    )


PROTOTYPE_COLUMNS = tuple(prototype_schema(1).names)


def prototype_table(
    prototypes: np.ndarray,
    class_ids: Sequence[str],
    support_counts: Sequence[int],
    payload: pd.DataFrame,
    provenance: PrototypeProvenance,
) -> pa.Table:
    """Rows sorted by ``class_id``; ``payload`` must hold exactly one row per class."""
    prototypes = np.asarray(prototypes, dtype=np.float32)
    if prototypes.ndim != 2 or prototypes.shape[0] != len(class_ids):
        raise ValueError(f"prototypes shape {prototypes.shape} for {len(class_ids)} class ids")
    if len(support_counts) != len(class_ids):
        raise ValueError(f"{len(support_counts)} support counts for {len(class_ids)} class ids")
    if payload["class_id"].duplicated().any():
        raise ValueError("payload holds more than one row for a class")
    indexed = payload.set_index("class_id")
    missing = sorted(set(class_ids) - set(indexed.index))
    if missing:
        raise ValueError(f"payload has no row for {len(missing)} class(es): {missing[:5]}")

    order = np.argsort(np.asarray(list(class_ids), dtype=object), kind="stable")
    sorted_ids = [class_ids[i] for i in order]
    rows = indexed.loc[sorted_ids, list(PAYLOAD_COLUMNS)]
    dim = int(prototypes.shape[1])

    def nullable_int(values: pd.Series) -> pa.Array:
        return pa.array([None if pd.isna(v) else int(v) for v in values], pa.int32())

    def nullable_str(values: pd.Series) -> pa.Array:
        return pa.array([None if pd.isna(v) else str(v) for v in values], pa.string())

    n = len(sorted_ids)
    columns = {
        "class_id": pa.array(sorted_ids, pa.string()),
        "embedding": pa.FixedSizeListArray.from_arrays(
            pa.array(prototypes[order].reshape(-1), pa.float32()), dim
        ),
        "support_count": pa.array([int(support_counts[i]) for i in order], pa.int32()),
        "make_id": nullable_int(rows["make_id"]),
        "make_name": nullable_str(rows["make_name"]),
        "model_id": nullable_int(rows["model_id"]),
        "model_name": nullable_str(rows["model_name"]),
        "year": nullable_int(rows["year"]),
        "generation_id": nullable_str(rows["generation_id"]),
        "mapping_status": nullable_str(rows["mapping_status"]),
        "prototype_method": pa.array([provenance.prototype_method] * n, pa.string()),
        "encoder": pa.array([provenance.encoder] * n, pa.string()),
        "encoder_revision": pa.array([provenance.encoder_revision] * n, pa.string()),
        "encoder_checksum": pa.array([provenance.encoder_checksum] * n, pa.string()),
        "preprocessing_id": pa.array([provenance.preprocessing_id] * n, pa.string()),
        "dimension": pa.array([dim] * n, pa.int32()),
        "run_id": pa.array([provenance.run_id] * n, pa.string()),
    }
    return pa.table(columns, schema=prototype_schema(dim))


class PrototypeTableOutputs(BaseModel):
    parquet: Path
    sha256: Path


def write_prototype_table(table: pa.Table, path: Path) -> PrototypeTableOutputs:
    """Write ``table`` to ``path`` and its SHA-256 to the ``.sha256`` sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    return PrototypeTableOutputs(parquet=path, sha256=write_sha256_sidecar(path))


def read_prototype_table(path: Path) -> tuple[pa.Table, np.ndarray]:
    """The stored table and its ``[C, dim]`` float32 prototype matrix in row order."""
    table = pq.read_table(path)
    return table, read_fixed_list_matrix(table, "embedding")
