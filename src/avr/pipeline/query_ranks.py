"""Per-query rank tables for one track and variant (issue M2-6).

Replays the gate's data path (:mod:`avr.pipeline.gate`) for the primary evaluation of each
track, Track A ``official_full`` (official query half against prototypes from the official
support half) and Track B ``test`` (seed split, prototypes from ``support``, queries
restricted to known-viewpoint rows), and hands the queries to
:func:`avr.evaluation.per_query.build_per_query`.

DUPLICATION NOTE: the loaders below (:func:`_read_checked`, :func:`_load_store`,
:func:`_track_a_rows`, :func:`_track_b_rows`) are copies of the private helpers of
``avr.pipeline.gate``, which was under review on another branch when this module was
written. Once that branch has landed, replace the copies with imports of the final helpers.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from avr.data.splits_io import read_sidecar_sha256
from avr.encoders.store import EmbeddingStore, StoreIdentity
from avr.evaluation.per_query import build_per_query
from avr.pipeline.e0 import class_id_of
from avr.pipeline.gate import TRACK_A_SPLIT_LABEL
from avr.prototypes.m0 import build_global_mean_prototypes

Track = Literal["track_a", "track_b"]
UNKNOWN_VIEWPOINT = -1
QUERY_PARTITION = {"track_a": "query", "track_b": "test"}


@dataclass(frozen=True)
class QueryRanksInputs:
    track: Track
    variant: str
    encoder: str
    embeddings_dir: Path
    manifest_path: Path
    split_path: Path


@dataclass
class QueryRanksResult:
    per_query: pd.DataFrame
    class_names: pd.DataFrame
    identity: StoreIdentity
    manifest_sha256: str
    split_sha256: str
    support_images: int
    prototype_count: int
    without_embedding: dict[str, int]
    queries_not_eligible: int


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_checked(path: Path) -> tuple[pd.DataFrame, str]:
    """A Parquet table whose ``.sha256`` sidecar must match (copy of the gate helper)."""
    digest = _sha256(path)
    sidecar = path.with_suffix(".sha256")
    if not sidecar.exists():
        raise ValueError(f"{path}: no .sha256 sidecar")
    if digest != read_sidecar_sha256(sidecar):
        raise ValueError(f"{path}: sha256 {digest} differs from the sidecar {sidecar.name}")
    table = pd.read_parquet(path)
    if table.image_id.duplicated().any():
        raise ValueError(f"{path}: split table lists an image id more than once")
    return table, digest


def _load_store(
    inputs: QueryRanksInputs,
) -> tuple[EmbeddingStore, dict[str, int], np.ndarray]:
    store = EmbeddingStore.from_directory(inputs.embeddings_dir, inputs.encoder, inputs.variant)
    ids, matrix = store.load_all()
    if not ids:
        raise ValueError(f"{store.directory}: no finished shards")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{store.directory}: duplicate image ids across shards")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("embeddings contain non-finite values")
    if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-3):
        raise ValueError("embeddings are not L2-normalised")
    return store, {image_id: row for row, image_id in enumerate(ids)}, matrix


def _track_a_rows(split: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    rows = split.merge(
        manifest[["image_id", "make_name", "model_name", "viewpoint", "validity_status"]],
        on="image_id",
        how="left",
        validate="one_to_one",
    )
    if rows.validity_status.isna().any():
        raise ValueError("split rows absent from the manifest")
    rows = rows[rows.validity_status == "valid"].copy()
    pairs = zip(rows.make_id, rows.model_id, strict=True)
    rows["class_id"] = [class_id_of(m, mo) for m, mo in pairs]
    rows["year"] = pd.array([pd.NA] * len(rows), dtype="Int16")
    return rows.sort_values("image_id").reset_index(drop=True)


def _track_b_rows(split: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    rows = split.merge(
        manifest[["image_id", "make_name", "model_name", "year", "validity_status"]],
        on="image_id",
        how="left",
        validate="one_to_one",
    )
    if rows.validity_status.isna().any():
        raise ValueError("split rows absent from the manifest")
    if (rows.validity_status != "valid").any():
        raise ValueError("Track B split table holds excluded manifest rows")
    if (rows.mapping_status != "provisional").any():
        raise ValueError("Track B rows are expected to be mapping_status = provisional")
    rows = rows.copy()
    rows["year"] = rows.year.astype("Int16")
    return rows.sort_values("image_id").reset_index(drop=True)


def _class_names(rows: pd.DataFrame) -> pd.DataFrame:
    columns = ["class_id", "make_name", "model_name", "year"]
    names = rows[columns].drop_duplicates()
    if names.class_id.duplicated().any():
        raise ValueError("a class maps to more than one make/model/year name payload")
    return names.sort_values("class_id").reset_index(drop=True)


def run_query_ranks(inputs: QueryRanksInputs) -> QueryRanksResult:
    manifest = pd.read_parquet(inputs.manifest_path)
    split, split_digest = _read_checked(inputs.split_path)
    store, row_of, matrix = _load_store(inputs)
    if inputs.track == "track_a":
        rows = _track_a_rows(split, manifest)
        support = rows[rows.split_track_a == "support"]
        query = rows[rows.split_track_a == "query"]
        split_label, mapping_status = TRACK_A_SPLIT_LABEL, "not_applicable"
        not_eligible = 0
    else:
        rows = _track_b_rows(split, manifest)
        support = rows[rows.partition == "support"]
        partition = rows[rows.partition == "test"]
        query = partition[partition.viewpoint != UNKNOWN_VIEWPOINT]
        not_eligible = len(partition) - len(query)
        split_label, mapping_status = f"{inputs.track}_{inputs.split_path.stem}", "provisional"
    support_all, query_all = len(support), len(query)
    support = support[support.image_id.isin(row_of)]
    query = query[query.image_id.isin(row_of)].reset_index(drop=True)
    overlap = set(support.image_id) & set(query.image_id)
    if overlap:
        raise ValueError(f"{len(overlap)} query image id(s) also listed as support")
    class_of = dict(zip(rows.image_id, rows.class_id, strict=True))
    support_ids = list(support.image_id)
    prototypes, class_ids = build_global_mean_prototypes(
        matrix[[row_of[i] for i in support_ids]],
        [class_of[i] for i in support_ids],
        support_ids,
        set(query.image_id),
    )
    meta = pd.DataFrame(
        {
            "image_id": query.image_id,
            "true_class_id": query.class_id,
            "make_id": query.make_id,
            "model_id": query.model_id,
            "year": query.year,
            "viewpoint": query.viewpoint,
        }
    )
    per_query = build_per_query(
        matrix[[row_of[i] for i in query.image_id]],
        prototypes,
        class_ids,
        meta,
        track=inputs.track,
        encoder=inputs.encoder,
        variant=inputs.variant,
        partition=QUERY_PARTITION[inputs.track],
        split_label=split_label,
        mapping_status=mapping_status,
        check_faiss=True,
    )
    return QueryRanksResult(
        per_query=per_query,
        class_names=_class_names(rows),
        identity=store.identity,
        manifest_sha256=_sha256(inputs.manifest_path),
        split_sha256=split_digest,
        support_images=len(support_ids),
        prototype_count=len(class_ids),
        without_embedding={
            "support": support_all - len(support_ids),
            "query": query_all - len(query),
        },
        queries_not_eligible=not_eligible,
    )
