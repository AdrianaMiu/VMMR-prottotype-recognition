"""E0 pipeline validation on the committed ten-class fixture (PROTOCOL.md §14 "Stage E0").

The fixture under ``tests/fixtures/e0_10class/`` holds a public manifest slice, frozen P1
embeddings for its images, the support/query assignment and the metric values recorded
when the fixture was built. :func:`run_e0` replays manifest validation, M0 prototypes,
exact search, ranking and classification metrics and the run manifest without a GPU or
torch, so CI can assert that none of those steps drifted.

The fixture split is NOT the Track A official split: :func:`select_fixture_split` picks
the classes with the most valid images and draws support and query images with a seeded
shuffle, keeping at most one image per duplicate group.
"""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from PIL import Image
from pydantic import BaseModel, ConfigDict

from avr.data.manifest import PUBLIC_COLUMNS, PUBLIC_SCHEMA
from avr.encoders.extract import Embedder
from avr.encoders.store import StoreIdentity, embedding_table, read_embedding_table
from avr.evaluation.classification import ClassificationMetrics, classification_metrics
from avr.evaluation.ranking import RankingMetrics, ranking_metrics
from avr.io import sha256_of_file
from avr.preprocessing.p3_batch import load_jobs
from avr.preprocessing.variants import prepare
from avr.prototypes.m0 import build_global_mean_prototypes
from avr.retrieval.exact import rank_by_cosine, rank_by_cosine_faiss
from avr.tracking.manifest import RunManifest, capture_environment
from avr.tracking.mlflow_run import REPO_ROOT, tracked_run

FIXTURE_SEED = 73
FIXTURE_CLASSES = 10
FIXTURE_SUPPORT = 5
FIXTURE_QUERY = 3


def class_id_of(make_id: int, model_id: int) -> str:
    """Make-model class id, zero-padded so lexicographic order equals numeric order."""
    return f"{int(make_id):03d}_{int(model_id):04d}"


class FixtureSplit(BaseModel):
    """Support and query image ids per class; lists are sorted, classes lexicographic."""

    model_config = ConfigDict(frozen=True)

    seed: int
    class_ids: list[str]
    support: dict[str, list[str]]
    query: dict[str, list[str]]

    @property
    def support_ids(self) -> list[str]:
        return [i for c in self.class_ids for i in self.support[c]]

    @property
    def query_ids(self) -> list[str]:
        return [i for c in self.class_ids for i in self.query[c]]


def select_fixture_split(
    rows: Iterable[Mapping],
    n_classes: int = FIXTURE_CLASSES,
    n_support: int = FIXTURE_SUPPORT,
    n_query: int = FIXTURE_QUERY,
    seed: int = FIXTURE_SEED,
) -> FixtureSplit:
    """Deterministic fixture split from public manifest rows.

    1. Keep valid rows; the class is ``class_id_of(make_id, model_id)``.
    2. Take the ``n_classes`` classes with the most valid images (ties by class id).
    3. Per class, in class-id order: sort the image ids, shuffle them with one
       ``numpy.random.default_rng(seed)`` shared across classes, walk the shuffled list
       skipping any image whose ``duplicate_group_id`` is already in the fixture, and
       assign the first ``n_support`` images to support and the next ``n_query`` to query.
    """
    valid = [r for r in rows if r["validity_status"] == "valid"]
    by_class: dict[str, list[Mapping]] = {}
    for r in valid:
        by_class.setdefault(class_id_of(r["make_id"], r["model_id"]), []).append(r)
    counts = Counter({c: len(v) for c, v in by_class.items()})
    ranked = sorted(counts, key=lambda c: (-counts[c], c))[:n_classes]
    class_ids = sorted(ranked)

    rng = np.random.default_rng(seed)
    seen_groups: set[str] = set()
    support: dict[str, list[str]] = {}
    query: dict[str, list[str]] = {}
    for class_id in class_ids:
        group_of = {
            r["image_id"]: r["duplicate_group_id"] or r["image_id"] for r in by_class[class_id]
        }
        image_ids = sorted(group_of)
        shuffled = [image_ids[i] for i in rng.permutation(len(image_ids))]
        chosen: list[str] = []
        for image_id in shuffled:
            if group_of[image_id] in seen_groups:
                continue
            seen_groups.add(group_of[image_id])
            chosen.append(image_id)
            if len(chosen) == n_support + n_query:
                break
        if len(chosen) < n_support + n_query:
            raise ValueError(
                f"class {class_id}: fewer than {n_support + n_query} group-distinct valid "
                f"images ({len(chosen)})"
            )
        support[class_id] = sorted(chosen[:n_support])
        query[class_id] = sorted(chosen[n_support:])
    return FixtureSplit(seed=seed, class_ids=class_ids, support=support, query=query)


# --- fixture files ---------------------------------------------------------------------

SLICE_NAME = "manifest_slice.parquet"
EMBEDDINGS_NAME = "embeddings.parquet"
SPLIT_NAME = "split.json"
EXPECTED_NAME = "expected_metrics.json"
README_NAME = "README.md"
SELECTION_RULE = (
    "fixture-only split, not the Track A official split: the classes with the most valid "
    "images, then per class a shuffle of the sorted image ids with numpy default_rng(seed), "
    "at most one image per duplicate_group_id, first n_support images as support and the "
    "next n_query as query"
)
PROTOTYPE_METHOD = "m0_global_mean"
DATASET_PROTOCOL = "compcars_web"


class FixtureSplitFile(FixtureSplit):
    """``split.json``: the split plus what it was drawn from and embedded with."""

    rule: str
    manifest_slice_sha256: str
    source_manifest_sha256: str
    embedding_identity: StoreIdentity
    git_commit: str


def write_fixture(
    fixture_dir: Path,
    manifest_slice: pa.Table,
    image_ids: Sequence[str],
    embeddings: np.ndarray,
    split: FixtureSplit,
    identity: StoreIdentity,
    source_manifest_sha256: str,
    git_commit: str,
) -> Path:
    """Write ``manifest_slice.parquet``, ``embeddings.parquet`` and ``split.json``."""
    fixture_dir = Path(fixture_dir)
    fixture_dir.mkdir(parents=True, exist_ok=True)
    if manifest_slice.schema.names != list(PUBLIC_COLUMNS):
        raise ValueError("manifest slice must carry exactly the public manifest columns")
    manifest_slice = manifest_slice.cast(PUBLIC_SCHEMA).sort_by("image_id")
    pq.write_table(manifest_slice, fixture_dir / SLICE_NAME, compression="zstd")
    order = np.argsort(np.asarray(image_ids, dtype=object), kind="stable")
    pq.write_table(
        embedding_table([image_ids[i] for i in order], np.asarray(embeddings)[order]),
        fixture_dir / EMBEDDINGS_NAME,
        compression="zstd",
    )
    split_file = FixtureSplitFile(
        **split.model_dump(),
        rule=SELECTION_RULE,
        manifest_slice_sha256=sha256_of_file(fixture_dir / SLICE_NAME),
        source_manifest_sha256=source_manifest_sha256,
        embedding_identity=identity,
        git_commit=git_commit,
    )
    (fixture_dir / SPLIT_NAME).write_text(
        json.dumps(split_file.model_dump(mode="json"), indent=2) + "\n"
    )
    return fixture_dir


# --- the pipeline ----------------------------------------------------------------------


def _validate_slice(fixture_dir: Path, split: FixtureSplitFile) -> dict[str, dict]:
    """Columns and checksum of the slice; returns valid rows by image id."""
    path = fixture_dir / SLICE_NAME
    digest = sha256_of_file(path)
    if digest != split.manifest_slice_sha256:
        raise ValueError(
            f"{path.name}: sha256 {digest} differs from split.json ({split.manifest_slice_sha256})"
        )
    table = pq.read_table(path)
    if table.schema.names != list(PUBLIC_COLUMNS):
        raise ValueError(f"{path.name}: columns {table.schema.names} are not the public columns")
    rows = {r["image_id"]: r for r in table.to_pylist()}
    if len(rows) != table.num_rows:
        raise ValueError(f"{path.name}: duplicate image ids")
    return rows


def _validate_split(split: FixtureSplitFile, rows: Mapping[str, Mapping]) -> None:
    support, query = split.support_ids, split.query_ids
    overlap = sorted(set(support) & set(query))
    if overlap:
        raise ValueError(f"{len(overlap)} query image id(s) also listed as support: {overlap[:5]}")
    for name, ids in (("support", support), ("query", query)):
        if len(set(ids)) != len(ids):
            raise ValueError(f"{name} image ids are not unique")
        missing = sorted(set(ids) - set(rows))
        if missing:
            raise ValueError(f"{name} image id(s) absent from the manifest slice: {missing[:5]}")
    for partition in (split.support, split.query):
        for class_id, ids in partition.items():
            for image_id in ids:
                row = rows[image_id]
                if row["validity_status"] != "valid":
                    raise ValueError(f"{image_id} is not a valid manifest row")
                if class_id_of(row["make_id"], row["model_id"]) != class_id:
                    raise ValueError(
                        f"{image_id} is listed under class {class_id} but belongs elsewhere"
                    )
    group_of = {i: rows[i]["duplicate_group_id"] or i for i in support + query}
    crossing = {group_of[i] for i in support} & {group_of[i] for i in query}
    if crossing:
        raise ValueError(f"duplicate group(s) cross support and query: {sorted(crossing)[:5]}")


def _load_embeddings(fixture_dir: Path, needed: Sequence[str]) -> dict[str, np.ndarray]:
    ids, matrix = read_embedding_table(fixture_dir / EMBEDDINGS_NAME)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("embeddings contain non-finite values")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("embeddings are not L2-normalised")
    by_id = dict(zip(ids, matrix, strict=True))
    missing = sorted(set(needed) - set(by_id))
    if missing:
        raise ValueError(f"embeddings missing for {len(missing)} image id(s): {missing[:5]}")
    return by_id


def _flatten(ranking: RankingMetrics, classification: ClassificationMetrics) -> dict[str, float]:
    metrics: dict[str, float] = {
        "query_count": ranking.query_count,
        "class_count": ranking.class_count,
        "tied_queries": ranking.tied_queries,
    }
    for k, value in ranking.top_k.items():
        metrics[f"top{k}"] = value
    for name in (
        "mrr",
        "mean_rank",
        "median_rank",
        "rank_p75",
        "rank_p90",
        "rank_p95",
        "mean_normalised_rank",
    ):
        metrics[name] = getattr(ranking, name)
    for name in (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "balanced_accuracy",
        "classes_without_predicted_positives",
    ):
        metrics[name] = getattr(classification, name)
    return metrics


def run_e0(fixture_dir: Path, tracking_dir: Path | None = None) -> dict[str, float]:
    """Validate the fixture, build M0 prototypes, rank, score and record a run manifest.

    ``tracking_dir`` is the MLflow file store (a temporary directory when ``None``). Runs
    on CPU with numpy and FAISS only; torch is never imported.
    """
    fixture_dir = Path(fixture_dir)
    split = FixtureSplitFile.model_validate_json((fixture_dir / SPLIT_NAME).read_text())
    rows = _validate_slice(fixture_dir, split)
    _validate_split(split, rows)
    support_ids, query_ids = split.support_ids, split.query_ids
    embedding_of = _load_embeddings(fixture_dir, support_ids + query_ids)

    def label(image_id: str) -> str:
        return class_id_of(rows[image_id]["make_id"], rows[image_id]["model_id"])

    support = np.stack([embedding_of[i] for i in support_ids])
    prototypes, class_ids = build_global_mean_prototypes(
        support, [label(i) for i in support_ids], support_ids, query_image_ids=set(query_ids)
    )
    queries = np.stack([embedding_of[i] for i in query_ids])
    ranking = rank_by_cosine_faiss(queries, prototypes, class_ids)
    matrix_ranking = rank_by_cosine(queries, prototypes, class_ids)
    if not np.array_equal(ranking.ranks, matrix_ranking.ranks):
        raise RuntimeError("FAISS exact search and matrix search disagree on the ranking")
    y_true = [label(i) for i in query_ids]
    y_pred = [class_ids[r] for r in ranking.ranks[:, 0]]
    metrics = _flatten(
        ranking_metrics(ranking.ranks, y_true, class_ids, tied_queries=ranking.tied_queries_at(1)),
        classification_metrics(y_true, y_pred, class_ids),
    )

    support_k = {len(v) for v in split.support.values()}
    manifest = RunManifest(
        run_id=f"e0_fixture_{fixture_dir.name}",
        dataset_protocol=DATASET_PROTOCOL,
        dataset_manifest_hash=split.source_manifest_sha256,
        split_manifest_hash=split.manifest_slice_sha256,
        generation_mapping_hash="not_applicable",
        encoder_id=split.embedding_identity.encoder,
        encoder_revision=split.embedding_identity.revision,
        encoder_checksum=split.embedding_identity.checkpoint_sha256,
        preprocessing_id=(
            f"{split.embedding_identity.preprocessing_id}+{split.embedding_identity.variant}"
        ),
        dimension_transform_id="native",
        prototype_method=PROTOTYPE_METHOD,
        support_k=support_k.pop() if len(support_k) == 1 else 0,
        seed=split.seed,
        **capture_environment(REPO_ROOT),
    )
    with ExitStack() as stack:
        if tracking_dir is None:
            tracking_dir = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        with tracked_run(manifest, tracking_dir=tracking_dir) as run:
            for name, value in metrics.items():
                run.log_metric(name, float(value))
    return metrics


# --- building the committed fixture ----------------------------------------------------


def _readme(
    split: FixtureSplitFile, metrics: Mapping[str, float], n_support: int, n_query: int
) -> str:
    identity = split.embedding_identity
    lines = [
        "# E0 ten-class fixture",
        "",
        'Frozen inputs for the E0 pipeline validation (PROTOCOL.md §14 "Stage E0"):',
        f"{len(split.class_ids)} make-model classes, {n_support} support and {n_query} query "
        "images per class, with their P1 embeddings. `avr.pipeline.e0.run_e0` replays manifest "
        "validation, M0 prototypes, exact search, ranking and classification metrics and the run "
        "manifest on CPU; `tests/integration/test_e0_fixture.py` asserts every metric equals "
        "`expected_metrics.json`.",
        "",
        "## This is a fixture split, not the Track A official split",
        "",
        "The Track A split (issue #9) was not available when the fixture was built. Classes and "
        "images were chosen by the rule in `split.json` (`rule`): " + split.rule + ".",
        f"Seed: {split.seed}. No duplicate group spans support and query.",
        "",
        "## Files",
        "",
        f"- `{SLICE_NAME}`: public manifest columns only (no paths) for the fixture images; "
        f"sha256 `{split.manifest_slice_sha256}`.",
        f"- `{EMBEDDINGS_NAME}`: `image_id` and L2-normalised float32 `embedding` rows.",
        f"- `{SPLIT_NAME}`: support/query image ids per class, selection rule, checksums and "
        "the embedding identity.",
        f"- `{EXPECTED_NAME}`: the metric values recorded when the fixture was built.",
        "",
        "## Provenance",
        "",
        f"- encoder: {identity.encoder} (revision {identity.revision}, checkpoint sha256 "
        f"{identity.checkpoint_sha256})",
        f"- preprocessing: {identity.preprocessing_id}, variant {identity.variant}, compute dtype "
        f"{identity.dtype}",
        f"- source manifest sha256: {split.source_manifest_sha256}",
        f"- built at commit {split.git_commit} by `scripts/build_e0_fixture.py`",
        "",
        "## Classes",
        "",
    ]
    lines += [f"- {class_id}" for class_id in split.class_ids]
    lines += ["", "## Recorded metrics", ""]
    lines += [f"- {name}: {value}" for name, value in metrics.items()]
    return "\n".join(lines) + "\n"


def build_fixture(
    fixture_dir: Path,
    manifest_path: Path,
    paths_path: Path,
    root: Path,
    embedder: Embedder,
    identity: StoreIdentity,
    git_commit: str,
    n_classes: int = FIXTURE_CLASSES,
    n_support: int = FIXTURE_SUPPORT,
    n_query: int = FIXTURE_QUERY,
    seed: int = FIXTURE_SEED,
    batch_size: int = 32,
    *,
    expand: float,
    tracking_dir: Path | None = None,
) -> dict[str, float]:
    """Select the split, embed its P1 crops, write the fixture and record its metrics.

    ``embedder`` is any object with ``embed(images) -> [n, dim]`` (a loaded
    :class:`avr.encoders.adapter.FrozenEncoder` in the build script); this module never
    imports torch. The returned metrics are also written to ``expected_metrics.json``.
    """
    fixture_dir = Path(fixture_dir)
    manifest = pq.read_table(manifest_path)
    split = select_fixture_split(
        manifest.to_pylist(), n_classes=n_classes, n_support=n_support, n_query=n_query, seed=seed
    )
    wanted = set(split.support_ids + split.query_ids)
    manifest_slice = manifest.filter(pc.is_in(manifest["image_id"], pa.array(sorted(wanted))))
    jobs = [j for j in load_jobs(manifest_path, paths_path) if j.image_id in wanted]
    if len(jobs) != len(wanted):
        raise ValueError(f"{len(wanted) - len(jobs)} fixture image(s) have no path table entry")

    blocks = []
    for start in range(0, len(jobs), batch_size):
        crops = []
        for job in jobs[start : start + batch_size]:
            with Image.open(Path(root) / job.relative_path) as image:
                crops.append(prepare(image.convert("RGB"), job.bbox_xyxy, "p1", expand=expand))
        blocks.append(embedder.embed(crops))

    write_fixture(
        fixture_dir,
        manifest_slice=manifest_slice.select(list(PUBLIC_COLUMNS)),
        image_ids=[j.image_id for j in jobs],
        embeddings=np.vstack(blocks),
        split=split,
        identity=identity,
        source_manifest_sha256=sha256_of_file(Path(manifest_path)),
        git_commit=git_commit,
    )
    metrics = run_e0(fixture_dir, tracking_dir=tracking_dir)
    (fixture_dir / EXPECTED_NAME).write_text(json.dumps(metrics, indent=2) + "\n")
    split_file = FixtureSplitFile.model_validate_json((fixture_dir / SPLIT_NAME).read_text())
    (fixture_dir / README_NAME).write_text(_readme(split_file, metrics, n_support, n_query))
    return metrics
