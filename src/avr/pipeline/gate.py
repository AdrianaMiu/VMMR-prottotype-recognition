"""M1 acceptance gate: one frozen encoder, M0 prototypes, exact search (issue M1-16).

:func:`run_track` replays, for one track and one preprocessing variant, what E0 does on
its fixture (:mod:`avr.pipeline.e0`) at full scale: it loads a finished embedding store,
joins the stored split table and the public manifest, builds M0 prototypes from support
rows only, ranks every query by exact search (matrix multiplication, asserted equal to
FAISS ``IndexFlatIP``), reports §13.1 and §13.2 metrics at every hierarchy level, persists
the prototypes (Parquet table and a Qdrant collection in embedded local mode, checked for
top-10 parity against the matrix ranking) and records a complete §17 run manifest.

Evaluations per track (PROTOCOL.md §5):

* Track A, ``official_full``: prototypes from every valid image of the official support
  half, queries from the official query half; ``validation_fit_only``: prototypes from the
  ``fit`` partition only, queries from the nested ``validation`` partition. Every output
  carries :data:`avr.data.split_identity.TRACK_A_LABEL`.
* Track B, ``test`` and ``validation``: prototypes from the ``support`` partition, queries
  from the named partition restricted to query-eligible rows (known viewpoint). Classes
  are the provisional model-year stand-ins of the split table, so every output is marked
  ``mapping_status = provisional``.

All numbers produced here are internal validation; none is an article result.
"""

from __future__ import annotations

import hashlib
import json
import resource
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from avr.data.split_identity import TRACK_A_LABEL
from avr.encoders.store import EmbeddingStore
from avr.evaluation.classification import ClassificationMetrics, classification_metrics
from avr.evaluation.hierarchy import error_decomposition, parent_ranking_metrics
from avr.evaluation.ranking import RankingMetrics, ranking_metrics
from avr.io import read_sidecar_sha256, repo_relative, sha256_of_file
from avr.pipeline.e0 import PROTOTYPE_METHOD, class_id_of
from avr.prototypes.m0 import build_global_mean_prototypes
from avr.prototypes.persist import (
    PrototypeProvenance,
    prototype_table,
    write_prototype_table,
)
from avr.retrieval.exact import Ranking, rank_by_cosine, rank_by_cosine_faiss
from avr.retrieval.qdrant_store import collection_name, export_prototypes, query_top_k
from avr.tracking.manifest import RunManifest, capture_environment
from avr.tracking.mlflow_run import REPO_ROOT, tracked_run

Track = Literal["track_a", "track_b"]

NOT_AN_ARTICLE_RESULT = (
    "internal validation only (M1 acceptance gate); not an article result and not to be "
    "cited in any article table"
)
DATASET_PROTOCOL = "compcars_web"
MAPPING_VERSION = {"track_a": "none", "track_b": "v0.1_provisional"}
PARITY_K = 10
PARITY_SCORE_TOLERANCE = 1e-5
UNKNOWN_VIEWPOINT = -1


class GateInputs(BaseModel):
    """Everything :func:`run_track` reads or writes, as paths and ids (no implicit defaults)."""

    model_config = ConfigDict(frozen=True)

    track: Track
    variant: str
    encoder: str
    embeddings_dir: Path
    manifest_path: Path
    split_path: Path
    mapping_queue_path: Path | None
    mask_metadata_path: Path | None = None
    output_dir: Path
    prototypes_dir: Path
    qdrant_dir: Path
    tracking_dir: Path
    seed: int = 73
    parity_queries: int = 1000

    @property
    def protocol_id(self) -> str:
        return f"{self.track}_{self.split_path.stem}"

    @property
    def run_id(self) -> str:
        return f"m1_gate_{self.track}_{self.variant}_{self.encoder}"


@dataclass
class GateResult:
    results: dict
    prototypes: np.ndarray
    class_ids: list[str]
    ranks: dict[str, np.ndarray] = field(default_factory=dict)
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)


# --- loading -----------------------------------------------------------------------------


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _read_checked(path: Path) -> tuple[pd.DataFrame, str]:
    """A Parquet table whose ``.sha256`` sidecar must match (fail closed, never silently).

    Used for the image manifest, the split table and the P3 mask metadata alike; every
    one of them lists each ``image_id`` at most once.
    """
    digest = sha256_of_file(path)
    sidecar = path.with_suffix(".sha256")
    if not sidecar.exists():
        raise ValueError(f"{path}: no .sha256 sidecar")
    if digest != read_sidecar_sha256(sidecar):
        raise ValueError(f"{path}: sha256 {digest} differs from the sidecar {sidecar.name}")
    table = pd.read_parquet(path)
    if table.image_id.duplicated().any():
        raise ValueError(f"{path}: table lists an image id more than once")
    return table, digest


def _mask_metadata_digest(inputs: GateInputs) -> str:
    """Verified digest of the P3 mask metadata the store was built from; p1 has none."""
    if inputs.variant != "p3":
        return "not_applicable"
    if inputs.mask_metadata_path is None:
        raise ValueError("variant 'p3' needs mask_metadata_path (data/manifests/p3_mask_metadata)")
    return _read_checked(inputs.mask_metadata_path)[1]


def _load_store(inputs: GateInputs) -> tuple[EmbeddingStore, dict[str, int], np.ndarray]:
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


@dataclass(frozen=True)
class Evaluation:
    """One support/query pairing: image ids, class labels and bookkeeping counts."""

    name: str
    support_ids: list[str]
    query_ids: list[str]
    without_embedding: dict[str, int]
    queries_not_eligible: int


def _drop_without_embedding(ids: Sequence[str], row_of: Mapping[str, int]) -> tuple[list, int]:
    kept = [i for i in ids if i in row_of]
    return kept, len(ids) - len(kept)


def _track_a_rows(split: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    rows = split.merge(
        manifest[["image_id", "make_name", "model_name", "validity_status"]],
        on="image_id",
        how="left",
        validate="one_to_one",
    )
    if rows.validity_status.isna().any():
        raise ValueError("split rows absent from the manifest")
    rows = rows[rows.validity_status == "valid"].copy()
    pairs = zip(rows.make_id, rows.model_id, strict=True)
    rows["class_id"] = [class_id_of(m, mo) for m, mo in pairs]
    rows["year"] = None
    rows["generation_id"] = None
    rows["mapping_status"] = "not_applicable"
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
    rows["generation_id"] = rows["class_id"]
    return rows.sort_values("image_id").reset_index(drop=True)


def _evaluations(
    inputs: GateInputs, rows: pd.DataFrame, row_of: Mapping[str, int]
) -> list[Evaluation]:
    def make(name: str, support: pd.DataFrame, query: pd.DataFrame, ineligible: int) -> Evaluation:
        support_ids, s_missing = _drop_without_embedding(list(support.image_id), row_of)
        query_ids, q_missing = _drop_without_embedding(list(query.image_id), row_of)
        overlap = set(support_ids) & set(query_ids)
        if overlap:
            raise ValueError(
                f"{name}: {len(overlap)} query image id(s) also listed as support: "
                f"{sorted(overlap)[:5]}"
            )
        return Evaluation(
            name=name,
            support_ids=support_ids,
            query_ids=query_ids,
            without_embedding={"support": s_missing, "query": q_missing},
            queries_not_eligible=ineligible,
        )

    if inputs.track == "track_a":
        support = rows[rows.split_track_a == "support"]
        return [
            make("official_full", support, rows[rows.split_track_a == "query"], 0),
            make(
                "validation_fit_only",
                support[support.track_a_validation == "fit"],
                support[support.track_a_validation == "validation"],
                0,
            ),
        ]
    support = rows[rows.partition == "support"]
    evaluations = []
    for name in ("test", "validation"):
        partition = rows[rows.partition == name]
        eligible = partition[partition.viewpoint != UNKNOWN_VIEWPOINT]
        evaluations.append(make(name, support, eligible, len(partition) - len(eligible)))
    return evaluations


# --- metrics -------------------------------------------------------------------------------


def _ranking_dict(m: RankingMetrics) -> dict:
    return {
        "query_count": m.query_count,
        "class_count": m.class_count,
        "tied_queries": m.tied_queries,
        "top_k": {str(k): v for k, v in m.top_k.items()},
        "mrr": m.mrr,
        "mean_rank": m.mean_rank,
        "median_rank": m.median_rank,
        "rank_p75": m.rank_p75,
        "rank_p90": m.rank_p90,
        "rank_p95": m.rank_p95,
        "mean_normalised_rank": m.mean_normalised_rank,
    }


def _classification_dict(m: ClassificationMetrics) -> dict:
    return {
        "class_count": len(m.class_ids),
        "query_count": m.query_count,
        "accuracy": m.accuracy,
        "macro_precision": m.macro_precision,
        "macro_recall": m.macro_recall,
        "macro_f1": m.macro_f1,
        "weighted_precision": m.weighted_precision,
        "weighted_recall": m.weighted_recall,
        "weighted_f1": m.weighted_f1,
        "balanced_accuracy": m.balanced_accuracy,
        "classes_without_predicted_positives": m.classes_without_predicted_positives,
        "per_class_recall": m.per_class_recall,
        "per_class_f1": m.per_class_f1,
    }


def _levels(track: Track) -> dict[str, Callable[[str], str] | None]:
    """Hierarchy levels as functions of the fine class id; ``None`` is the fine level."""
    if track == "track_a":
        return {"make_model": None, "make": lambda c: c.split("_")[0]}
    return {
        "class": None,
        "model": lambda c: "/".join(c.split("/")[:2]),
        "make": lambda c: c.split("/")[0],
    }


def _make_model_of(track: Track) -> Callable[[str], tuple[str, str]]:
    if track == "track_a":
        return lambda c: (c.split("_")[0], c)
    return lambda c: (c.split("/")[0], "/".join(c.split("/")[:2]))


def _level_metrics(
    track: Track, ranking: Ranking, y_true: list[str], class_ids: list[str]
) -> dict[str, dict]:
    levels: dict[str, dict] = {}
    tied = ranking.tied_queries_at(1)
    y_pred = [class_ids[r] for r in ranking.ranks[:, 0]]
    for name, parent in _levels(track).items():
        if parent is None:
            rank_m = ranking_metrics(ranking.ranks, y_true, class_ids, tied_queries=tied)
            class_m = classification_metrics(y_true, y_pred, class_ids)
        else:
            parent_of = {c: parent(c) for c in class_ids}
            parents = sorted(set(parent_of.values()))
            rank_m = parent_ranking_metrics(
                ranking.ranks, [parent(c) for c in y_true], class_ids, parent_of, tied_queries=tied
            )
            class_m = classification_metrics(
                [parent(c) for c in y_true], [parent(c) for c in y_pred], parents
            )
        levels[name] = {
            "ranking": _ranking_dict(rank_m),
            "classification": _classification_dict(class_m),
        }
    return levels


def _rank(queries: np.ndarray, prototypes: np.ndarray, class_ids: list[str]) -> Ranking:
    ranking = rank_by_cosine(queries, prototypes, class_ids)
    faiss_ranking = rank_by_cosine_faiss(queries, prototypes, class_ids)
    if not np.array_equal(ranking.ranks, faiss_ranking.ranks):
        raise RuntimeError("FAISS exact search and matrix search disagree on the ranking")
    return ranking


# --- persistence -------------------------------------------------------------------------


def _payload(rows: pd.DataFrame, class_ids: Sequence[str]) -> pd.DataFrame:
    columns = [
        "class_id",
        "make_id",
        "make_name",
        "model_id",
        "model_name",
        "year",
        "generation_id",
        "mapping_status",
    ]
    payload = rows[columns].drop_duplicates("class_id")
    conflicting = rows[columns].drop_duplicates()
    if len(conflicting) != len(payload):
        raise ValueError("a class maps to more than one make/model/year payload")
    return payload[payload.class_id.isin(set(class_ids))]


def _parity(
    inputs: GateInputs,
    collection: str,
    queries: np.ndarray,
    prototypes: np.ndarray,
    ranking: Ranking,
    class_ids: list[str],
) -> dict:
    """Qdrant exact top-k against the matrix top-k on the first ``parity_queries`` queries.

    The two backends sum float32 products in different orders (Qdrant also re-normalises
    on insert), so two prototypes whose cosine scores differ by less than
    ``PARITY_SCORE_TOLERANCE`` may swap places; such a position counts as equal within
    tolerance. Any other difference fails the gate.
    """
    n = min(inputs.parity_queries, queries.shape[0])
    k = min(PARITY_K, len(class_ids))
    hits = query_top_k(inputs.qdrant_dir, collection, queries[:n], k)
    qdrant_top = [[h.class_id for h in row] for row in hits]
    matrix_top = [[class_ids[r] for r in ranking.ranks[q, :k]] for q in range(n)]
    raw = np.ascontiguousarray(queries[:n], dtype=np.float32) @ prototypes.T
    index = {c: i for i, c in enumerate(class_ids)}
    ordered = within = mismatched = 0
    for q in range(n):
        if qdrant_top[q] == matrix_top[q]:
            ordered += 1
            continue
        gaps = [
            abs(float(raw[q, index[a]]) - float(raw[q, index[b]]))
            for a, b in zip(qdrant_top[q], matrix_top[q], strict=True)
            if a != b
        ]
        if len(qdrant_top[q]) == k and max(gaps) <= PARITY_SCORE_TOLERANCE:
            within += 1
        else:
            mismatched += 1
    if mismatched:
        raise RuntimeError(
            f"Qdrant exact top-{k} differs from the matrix top-{k} on {mismatched} of {n} "
            f"queries beyond the score tolerance {PARITY_SCORE_TOLERANCE}"
        )
    return {
        "queries": n,
        "k": k,
        "ordered_equal": ordered,
        "equal_within_tolerance": within,
        "score_tolerance": PARITY_SCORE_TOLERANCE,
        "passed": mismatched == 0,
    }


# --- the pipeline ----------------------------------------------------------------------------


def run_track(inputs: GateInputs) -> GateResult:
    started = time.perf_counter()
    environment = capture_environment(REPO_ROOT)
    manifest, manifest_digest = _read_checked(inputs.manifest_path)
    mask_metadata_digest = _mask_metadata_digest(inputs)
    split, split_digest = _read_checked(inputs.split_path)
    mapping_digest = (
        sha256_of_file(inputs.mapping_queue_path) if inputs.mapping_queue_path else "not_applicable"
    )
    store, row_of, matrix = _load_store(inputs)
    identity = store.identity
    rows = (
        _track_a_rows(split, manifest)
        if inputs.track == "track_a"
        else _track_b_rows(split, manifest)
    )
    class_of = dict(zip(rows.image_id, rows.class_id, strict=True))
    evaluations = _evaluations(inputs, rows, row_of)
    all_class_ids = sorted(set(rows.class_id))
    preprocessing_id = f"{identity.preprocessing_id}+{identity.variant}"

    result = GateResult(results={}, prototypes=np.zeros((0, 0), np.float32), class_ids=[])
    outputs: dict[str, dict] = {}
    determinism: dict[str, str] = {}
    primary = evaluations[0]
    for ev in evaluations:
        support = matrix[[row_of[i] for i in ev.support_ids]]
        queries = matrix[[row_of[i] for i in ev.query_ids]]
        prototypes, class_ids = build_global_mean_prototypes(
            support, [class_of[i] for i in ev.support_ids], ev.support_ids, set(ev.query_ids)
        )
        y_true = [class_of[i] for i in ev.query_ids]
        missing_classes = sorted(set(y_true) - set(class_ids))
        if missing_classes:
            raise ValueError(
                f"{ev.name}: {len(missing_classes)} query class(es) have no support prototype: "
                f"{missing_classes[:5]}"
            )
        ranking = _rank(queries, prototypes, class_ids)
        y_pred = [class_ids[r] for r in ranking.ranks[:, 0]]
        outputs[ev.name] = {
            "support_images": len(ev.support_ids),
            "query_images": len(ev.query_ids),
            "class_count": len(class_ids),
            "classes_in_split": len(all_class_ids),
            "images_without_embedding": ev.without_embedding,
            "queries_not_eligible_unknown_viewpoint": ev.queries_not_eligible,
            "tied_queries_top1": ranking.tied_queries_at(1),
            "levels": _level_metrics(inputs.track, ranking, y_true, class_ids),
            "error_decomposition": error_decomposition(
                y_true, y_pred, _make_model_of(inputs.track)
            ).as_dict(),
        }
        ranks = ranking.ranks.astype(np.int32)
        result.ranks[ev.name] = ranks
        result.embeddings[ev.name] = np.vstack([support, queries])
        determinism[f"{ev.name}/prototypes_sha256"] = _sha256_array(prototypes)
        determinism[f"{ev.name}/ranks_sha256"] = _sha256_array(ranks)
        determinism[f"{ev.name}/embeddings_sha256"] = _sha256_array(result.embeddings[ev.name])
        if ev is primary:
            result.prototypes, result.class_ids = prototypes, class_ids
            primary_support_counts = pd.Series([class_of[i] for i in ev.support_ids]).value_counts()
            primary_queries = queries
            primary_ranking = ranking

    provenance = PrototypeProvenance(
        prototype_method=PROTOTYPE_METHOD,
        encoder=identity.encoder,
        encoder_revision=identity.revision,
        encoder_checksum=identity.checkpoint_sha256,
        preprocessing_id=preprocessing_id,
        run_id=inputs.run_id,
    )
    table = prototype_table(
        result.prototypes,
        result.class_ids,
        [int(primary_support_counts[c]) for c in result.class_ids],
        _payload(rows, result.class_ids),
        provenance,
    )
    parquet_path = (
        inputs.prototypes_dir
        / identity.encoder
        / identity.variant
        / f"m0_{inputs.protocol_id}.parquet"
    )
    written = write_prototype_table(table, parquet_path)
    collection = collection_name(
        inputs.protocol_id,
        MAPPING_VERSION[inputs.track],
        identity.encoder,
        preprocessing_id,
        int(result.prototypes.shape[1]),
        PROTOTYPE_METHOD,
    )
    export = export_prototypes(inputs.qdrant_dir, collection, table)
    parity = _parity(
        inputs, collection, primary_queries, result.prototypes, primary_ranking, result.class_ids
    )

    manifest_record = RunManifest(
        run_id=inputs.run_id,
        dataset_protocol=f"{DATASET_PROTOCOL}:{inputs.protocol_id}",
        dataset_manifest_hash=manifest_digest,
        split_manifest_hash=split_digest,
        generation_mapping_hash=mapping_digest,
        encoder_id=identity.encoder,
        encoder_revision=identity.revision,
        encoder_checksum=identity.checkpoint_sha256,
        preprocessing_id=preprocessing_id,
        dimension_transform_id="native",
        prototype_method=PROTOTYPE_METHOD,
        support_k="full",
        seed=inputs.seed,
        **environment,
    )
    with tracked_run(manifest_record, tracking_dir=inputs.tracking_dir) as run:
        for ev_name, ev_out in outputs.items():
            for level, metrics in ev_out["levels"].items():
                for k, v in metrics["ranking"]["top_k"].items():
                    run.log_metric(f"{ev_name}/{level}/top{k}", v)
                run.log_metric(f"{ev_name}/{level}/mrr", metrics["ranking"]["mrr"])
                run.log_metric(f"{ev_name}/{level}/macro_f1", metrics["classification"]["macro_f1"])
                run.log_metric(f"{ev_name}/{level}/accuracy", metrics["classification"]["accuracy"])
        run.log_param("mask_metadata_sha256", mask_metadata_digest)
        run.log_metric("qdrant_parity_queries", parity["queries"])

    elapsed = time.perf_counter() - started
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    result.results = {
        "status": NOT_AN_ARTICLE_RESULT,
        "article_result": False,
        "track": inputs.track,
        "variant": inputs.variant,
        "protocol_id": inputs.protocol_id,
        "split_label": TRACK_A_LABEL if inputs.track == "track_a" else inputs.protocol_id,
        "mapping_status": "not_applicable" if inputs.track == "track_a" else "provisional",
        "mapping_version": MAPPING_VERSION[inputs.track],
        "mask_metadata_sha256": mask_metadata_digest,
        "embedding_store": {
            **identity.model_dump(),
            "directory": repo_relative(store.directory),
            "rows": int(matrix.shape[0]),
            "dimension": int(matrix.shape[1]),
            "shards": len(store.finished_shards()),
        },
        "evaluations": outputs,
        "prototypes": {
            "parquet": repo_relative(written.parquet),
            "parquet_sha256": sha256_of_file(written.parquet),
            "class_count": table.num_rows,
            "support_counts": {
                "min": int(primary_support_counts.min()),
                "max": int(primary_support_counts.max()),
                "total": int(primary_support_counts.sum()),
            },
        },
        "qdrant": {
            "path": repo_relative(inputs.qdrant_dir),
            **export.model_dump(),
            "parity": parity,
        },
        "determinism": determinism,
        "run_manifest": json.loads(manifest_record.model_dump_json()),
        "runtime_s": round(elapsed, 2),
        "peak_rss_mb": round(peak_rss_mb, 1),
    }
    inputs.output_dir.mkdir(parents=True, exist_ok=True)
    (inputs.output_dir / f"{inputs.track}_{inputs.variant}.json").write_text(
        json.dumps(result.results, indent=2) + "\n"
    )
    return result
