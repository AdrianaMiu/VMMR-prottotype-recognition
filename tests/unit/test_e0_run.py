"""``run_e0`` on a hand-built two-class fixture (PROTOCOL.md §14 "Stage E0", §23).

Expected values are worked out by hand in the comments; nothing is recomputed with the
code under test.
"""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pytest

from avr.data.manifest import PUBLIC_SCHEMA
from avr.encoders.store import StoreIdentity
from avr.pipeline.e0 import FixtureSplit, run_e0, write_fixture

IDENTITY = StoreIdentity(
    encoder="toy",
    revision="r1",
    checkpoint_sha256="0" * 64,
    preprocessing_id="toy_pre",
    variant="p1",
    dtype="float32",
)


def manifest_row(image_id: str, make: int, model: int, group: str | None = None) -> dict:
    return {
        "image_id": image_id,
        "sha256": image_id * 4,
        "width": 10,
        "height": 10,
        "make_id": make,
        "make_name": f"make{make}",
        "model_id": model,
        "model_name": f"model{model}",
        "year": 2010,
        "viewpoint": 1,
        "bbox_xyxy": [0, 0, 9, 9],
        "duplicate_group_id": group or image_id,
        "source_group_id": group or image_id,
        "validity_status": "valid",
    }


def unit(x: float, y: float) -> list[float]:
    n = (x * x + y * y) ** 0.5
    return [x / n, y / n]


@pytest.fixture
def toy_fixture(tmp_path):
    ids = ["a1", "a2", "aq1", "aq2", "b1", "b2", "bq1"]
    rows = [
        manifest_row(i, 1 if i.startswith("a") else 2, 1 if i.startswith("a") else 2) for i in ids
    ]
    table = pa.Table.from_pylist(rows, schema=PUBLIC_SCHEMA)
    # support A: (1,0), (0.6,0.8) -> mean (0.8,0.4); support B: (0,1), (-0.6,0.8) -> (-0.3,0.9)
    # queries: aq1=(1,0) -> A; aq2=(0,1) -> B (wrong, true rank 2); bq1=(0,1) -> B
    embeddings = np.array(
        [[1, 0], [0.6, 0.8], [1, 0], [0, 1], [0, 1], [-0.6, 0.8], unit(-0.1, 1)], dtype=np.float32
    )
    split = FixtureSplit(
        seed=73,
        class_ids=["001_0001", "002_0002"],
        support={"001_0001": ["a1", "a2"], "002_0002": ["b1", "b2"]},
        query={"001_0001": ["aq1", "aq2"], "002_0002": ["bq1"]},
    )
    fixture_dir = tmp_path / "fixture"
    write_fixture(
        fixture_dir,
        manifest_slice=table,
        image_ids=ids,
        embeddings=embeddings,
        split=split,
        identity=IDENTITY,
        source_manifest_sha256="f" * 64,
        git_commit="deadbeef",
    )
    return fixture_dir


def test_metrics_match_the_hand_calculation(toy_fixture, tmp_path):
    metrics = run_e0(toy_fixture, tracking_dir=tmp_path / "mlruns")

    expected = {
        "query_count": 3,
        "class_count": 2,
        "tied_queries": 0,
        "top1": 2 / 3,
        "top5": 1.0,
        "top10": 1.0,
        "top20": 1.0,
        "mrr": (1 + 0.5 + 1) / 3,
        "mean_rank": 4 / 3,
        "median_rank": 1.0,
        "rank_p75": 1.5,
        "rank_p90": 1.8,
        "rank_p95": 1.9,
        "mean_normalised_rank": 1 / 3,
        "accuracy": 2 / 3,
        "macro_precision": 0.75,
        "macro_recall": 0.75,
        "macro_f1": 2 / 3,
        "weighted_precision": 1 * 2 / 3 + 0.5 / 3,
        "weighted_recall": 0.5 * 2 / 3 + 1 / 3,
        "weighted_f1": 2 / 3,
        "balanced_accuracy": 0.75,
        "classes_without_predicted_positives": 0,
    }
    assert set(metrics) == set(expected)
    for key, value in expected.items():
        assert metrics[key] == pytest.approx(value, abs=1e-9), key


def test_run_writes_a_complete_run_manifest_with_the_fixture_hashes(toy_fixture, tmp_path):
    import mlflow

    tracking_dir = tmp_path / "mlruns"
    run_e0(toy_fixture, tracking_dir=tracking_dir)

    client = mlflow.MlflowClient(tracking_uri=tracking_dir.as_uri())
    experiment = client.get_experiment_by_name("avr")
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    params = runs[0].data.params
    split = json.loads((toy_fixture / "split.json").read_text())
    assert params["dataset_manifest_hash"] == "f" * 64
    assert params["split_manifest_hash"] == split["manifest_slice_sha256"]
    assert params["encoder_id"] == "toy"
    assert params["encoder_checksum"] == "0" * 64
    assert params["preprocessing_id"] == "toy_pre+p1"
    assert params["prototype_method"] == "m0_global_mean"
    assert params["support_k"] == "2"
    assert params["seed"] == "73"
    assert runs[0].data.metrics["top1"] == pytest.approx(2 / 3)


def test_tampered_manifest_slice_is_rejected(toy_fixture, tmp_path):
    path = toy_fixture / "manifest_slice.parquet"
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="sha256"):
        run_e0(toy_fixture, tracking_dir=tmp_path / "mlruns")


def test_query_image_among_support_is_rejected(toy_fixture, tmp_path):
    split_path = toy_fixture / "split.json"
    split = json.loads(split_path.read_text())
    split["support"]["001_0001"].append("bq1")
    split_path.write_text(json.dumps(split))
    with pytest.raises(ValueError, match="query image"):
        run_e0(toy_fixture, tracking_dir=tmp_path / "mlruns")
