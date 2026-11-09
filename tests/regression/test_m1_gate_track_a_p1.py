"""Track A P1 gate metrics must reproduce the committed ``reports/m1_gate/track_a_p1.json``.

Runs only where the public image manifest and the finished DINOv3 ViT-B/16 P1 embedding
store are present (``AVR_EMBEDDINGS_DIR`` overrides ``data/embeddings``); prototypes, the
Qdrant store and the MLflow store go to a temporary directory so the committed sidecars
are never touched. The checks on the committed JSON itself run everywhere. The numbers are
internal validation only (PROTOCOL.md §25 M1), never article results.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from avr.config import CONFIG_DIR, load_config
from avr.data.manifest import MANIFEST_NAME
from avr.pipeline.gate import TRACK_A_SPLIT_LABEL, GateInputs, run_track

REPO_ROOT = CONFIG_DIR.parent
REPORT = REPO_ROOT / "reports" / "m1_gate" / "track_a_p1.json"
MANIFEST = REPO_ROOT / "data" / "manifests" / f"{MANIFEST_NAME}.parquet"
GATE_CFG = load_config("experiments/m1_gate").experiments
ENCODER = "dinov3_vitb16"
EMBEDDINGS_DIR = Path(os.environ.get("AVR_EMBEDDINGS_DIR", REPO_ROOT / "data" / "embeddings"))
STORE = EMBEDDINGS_DIR / ENCODER / "p1"
TOLERANCE = 1e-9


def recorded() -> dict:
    return json.loads(REPORT.read_text())


def test_committed_report_is_labelled_as_internal_validation_on_the_official_split():
    report = recorded()
    assert report["article_result"] is False
    assert "not an article result" in report["status"]
    assert report["split_label"] == TRACK_A_SPLIT_LABEL
    assert report["mapping_status"] == "not_applicable"
    assert report["evaluations"]["official_full"]["class_count"] == 431
    assert report["rerun"]["comparisons"][0]["prototypes_byte_identical"] is True
    assert all(report["rerun"]["comparisons"][0]["ranks_byte_identical"].values())


def _assert_close(expected, actual, path: str) -> None:
    if isinstance(expected, dict):
        assert set(expected) == set(actual), path
        for key in expected:
            _assert_close(expected[key], actual[key], f"{path}/{key}")
    elif isinstance(expected, float) or isinstance(actual, float):
        assert math.isclose(expected, actual, abs_tol=TOLERANCE), (path, expected, actual)
    else:
        assert expected == actual, (path, expected, actual)


@pytest.mark.skipif(
    not (MANIFEST.exists() and STORE.exists()),
    reason="image manifest or DINOv3 P1 embedding store not present",
)
def test_rerun_reproduces_the_committed_metrics(tmp_path: Path):
    report = recorded()
    shards = len(list(STORE.glob("shard_*.json")))
    recorded_shards = report["embedding_store"]["shards"]
    if shards < recorded_shards:
        pytest.skip(f"P1 store has {shards} shards, the report used {recorded_shards}")

    inputs = GateInputs(
        track="track_a",
        variant="p1",
        encoder=ENCODER,
        embeddings_dir=EMBEDDINGS_DIR,
        manifest_path=MANIFEST,
        split_path=REPO_ROOT / str(GATE_CFG.splits.track_a),
        mapping_queue_path=None,
        output_dir=tmp_path / "reports",
        prototypes_dir=tmp_path / "prototypes",
        qdrant_dir=tmp_path / "qdrant",
        tracking_dir=tmp_path / "mlruns",
        seed=73,
        parity_queries=int(GATE_CFG.parity_queries),
    )
    result = run_track(inputs)

    _assert_close(report["evaluations"], result.results["evaluations"], "evaluations")
    assert result.results["determinism"] == report["determinism"]
    assert result.results["prototypes"]["parquet_sha256"] == report["prototypes"]["parquet_sha256"]
    assert result.results["qdrant"]["parity"]["passed"] is True
