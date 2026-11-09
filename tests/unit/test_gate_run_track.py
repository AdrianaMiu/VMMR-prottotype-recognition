"""``run_track`` on hand-built three-class stores (issue M1-16, PROTOCOL.md §5, §11-§13).

No torch, no dataset: the manifest slice, the split tables and the embedding store are
written into ``tmp_path`` and every expected value is read off the vectors in the comments.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from avr.data.manifest import PUBLIC_SCHEMA
from avr.data.split_identity import TRACK_A_LABEL
from avr.data.split_track_a import PUBLIC_SCHEMA as TRACK_A_SCHEMA
from avr.data.split_track_b import OUTPUT_SCHEMA as TRACK_B_SCHEMA
from avr.data.splits_io import write_split_manifest
from avr.encoders.store import EmbeddingStore, StoreIdentity
from avr.pipeline.gate import GateInputs, run_track
from avr.prototypes.persist import read_prototype_table

IDENTITY = StoreIdentity(
    encoder="toy",
    revision="r1",
    checkpoint_sha256="0" * 64,
    preprocessing_id="toy_pre",
    variant="p1",
    dtype="float32",
)
X, Y, Z = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]


def unit(*values: float) -> list[float]:
    norm = sum(v * v for v in values) ** 0.5
    return [v / norm for v in values]


def manifest_row(image_id: str, make: int, model: int, year: int, viewpoint: int = 1) -> dict:
    return {
        "image_id": image_id,
        "sha256": image_id * 4,
        "width": 10,
        "height": 10,
        "make_id": make,
        "make_name": f"make{make}",
        "model_id": model,
        "model_name": f"model{model}",
        "year": year,
        "viewpoint": viewpoint,
        "bbox_xyxy": [0, 0, 9, 9],
        "duplicate_group_id": image_id,
        "source_group_id": image_id,
        "validity_status": "valid",
    }


def write_store(root: Path, vectors: dict[str, list[float]]) -> None:
    ids = sorted(vectors)
    EmbeddingStore(root, IDENTITY).write_shard(
        0, ids, np.array([vectors[i] for i in ids], dtype=np.float32)
    )


@pytest.fixture
def common(tmp_path: Path) -> dict:
    queue = tmp_path / "queue.csv"
    queue.write_text("make_id,model_id\n1,1\n")
    return {
        "tmp": tmp_path,
        "queue": queue,
        "output_dir": tmp_path / "reports",
        "prototypes_dir": tmp_path / "prototypes",
        "qdrant_dir": tmp_path / "qdrant",
        "tracking_dir": tmp_path / "mlruns",
    }


# --- Track A ---------------------------------------------------------------------------


@pytest.fixture
def track_a(common: dict) -> GateInputs:
    tmp = common["tmp"]
    # classes: A = 001_0001 (make 1), B = 001_0002 (make 1), C = 002_0003 (make 2)
    rows = [
        manifest_row("a1", 1, 1, 2010),
        manifest_row("a2", 1, 1, 2011),
        manifest_row("b1", 1, 2, 2010),
        manifest_row("b2", 1, 2, 2010),
        manifest_row("c1", 2, 3, 2014),
        manifest_row("c2", 2, 3, 2014),
        manifest_row("qa1", 1, 1, 2010),
        manifest_row("qa2", 1, 1, 2010),
        manifest_row("qc1", 2, 3, 2014),
        manifest_row("qx", 1, 1, 2010),  # excluded row: never a query
        manifest_row("qm", 2, 3, 2014),  # valid query row without an embedding (P3 missing)
    ]
    rows[-2]["validity_status"] = "excluded"
    rows[-2]["exclusion_reason"] = "degenerate_bbox"
    manifest = tmp / "manifest.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=PUBLIC_SCHEMA), manifest)

    def split_row(image_id: str, make: int, model: int, side: str, part: str | None) -> dict:
        return {
            "image_id": image_id,
            "sha256": image_id * 4,
            "make_id": make,
            "model_id": model,
            "split_track_a": side,
            "track_a_validation": part,
        }

    split = pa.Table.from_pylist(
        [
            split_row("a1", 1, 1, "support", "fit"),
            split_row("a2", 1, 1, "support", "validation"),
            split_row("b1", 1, 2, "support", "fit"),
            split_row("b2", 1, 2, "support", "validation"),
            split_row("c1", 2, 3, "support", "fit"),
            split_row("c2", 2, 3, "support", "validation"),
            split_row("qa1", 1, 1, "query", None),
            split_row("qa2", 1, 1, "query", None),
            split_row("qc1", 2, 3, "query", None),
            split_row("qx", 1, 1, "query", None),
            split_row("qm", 2, 3, "query", None),
        ],
        schema=TRACK_A_SCHEMA,
    )
    split_path = write_split_manifest(split, tmp / "track_a" / "official_cvpr2015.parquet").manifest
    # support: A on x, B on y, C on z (fit and validation copies identical).
    # queries: qa1 = x -> A (rank 1); qa2 = (0, .9, .1) -> B (A at rank 3, make 1 at rank 1);
    # qc1 = (.1, 0, .9) -> C (rank 1). qm has no embedding; qx is excluded.
    write_store(
        tmp / "emb",
        {
            "a1": X,
            "a2": X,
            "b1": Y,
            "b2": Y,
            "c1": Z,
            "c2": Z,
            "qa1": X,
            "qa2": unit(0.0, 0.9, 0.1),
            "qc1": unit(0.1, 0.0, 0.9),
            "qx": X,
        },
    )
    return GateInputs(
        track="track_a",
        variant="p1",
        encoder="toy",
        embeddings_dir=tmp / "emb",
        manifest_path=manifest,
        split_path=split_path,
        mapping_queue_path=None,
        seed=73,
        parity_queries=3,
        **{k: v for k, v in common.items() if k.endswith("_dir")},
    )


def test_track_a_metrics_match_the_hand_calculation(track_a: GateInputs):
    result = run_track(track_a)
    r = result.results

    assert r["article_result"] is False
    assert r["split_label"] == TRACK_A_LABEL
    assert not r["split_label"].endswith(".")
    assert r["mapping_status"] == "not_applicable"
    full = r["evaluations"]["official_full"]
    assert full["support_images"] == 6
    assert full["query_images"] == 3
    assert full["class_count"] == 3
    assert full["images_without_embedding"] == {"support": 0, "query": 1}
    mm = full["levels"]["make_model"]
    assert mm["ranking"]["top_k"]["1"] == pytest.approx(2 / 3)
    assert mm["ranking"]["top_k"]["5"] == 1.0
    assert mm["ranking"]["mrr"] == pytest.approx((1 + 1 / 3 + 1) / 3)
    assert mm["ranking"]["tied_queries"] == 0
    assert mm["classification"]["accuracy"] == pytest.approx(2 / 3)
    # per class: A recall 1/2, B recall 0 (no queries), C recall 1 -> macro recall 1/2
    assert mm["classification"]["macro_recall"] == pytest.approx(1 / 2)
    assert mm["classification"]["per_class_recall"]["001_0002"] == 0.0
    make = full["levels"]["make"]
    assert make["ranking"]["top_k"]["1"] == 1.0
    assert make["classification"]["accuracy"] == 1.0
    assert make["classification"]["class_count"] == 2
    d = full["error_decomposition"]
    assert d["wrong_model_same_make"] == 1 and d["wrong_make"] == 0
    assert d["wrong_generation_same_model"] == 0

    fit = r["evaluations"]["validation_fit_only"]
    assert fit["support_images"] == 3 and fit["query_images"] == 3
    assert fit["levels"]["make_model"]["ranking"]["top_k"]["1"] == 1.0


def test_track_a_persists_prototypes_qdrant_and_run_manifest(track_a: GateInputs):
    import mlflow

    result = run_track(track_a)
    r = result.results

    parquet = track_a.prototypes_dir / "toy" / "p1" / "m0_track_a_official_cvpr2015.parquet"
    assert parquet.exists()
    digest = hashlib.sha256(parquet.read_bytes()).hexdigest()
    assert parquet.with_suffix(".sha256").read_text().split()[0] == digest
    assert r["prototypes"]["parquet_sha256"] == digest
    table, matrix = read_prototype_table(parquet)
    assert table.column("class_id").to_pylist() == ["001_0001", "001_0002", "002_0003"]
    assert table.column("support_count").to_pylist() == [2, 2, 2]
    assert table.column("make_name").to_pylist() == ["make1", "make1", "make2"]
    assert table.column("year").to_pylist() == [None, None, None]
    assert table.column("mapping_status").to_pylist() == ["not_applicable"] * 3
    np.testing.assert_array_equal(matrix, np.eye(3, dtype=np.float32))

    q = r["qdrant"]
    assert q["collection"] == (
        "track_a_official_cvpr2015__none__toy__toy_pre+p1__3__m0_global_mean"
    )
    assert q["point_count"] == 3
    assert q["parity"] == {
        "queries": 3,
        "k": 3,
        "ordered_equal": 3,
        "equal_within_tolerance": 0,
        "score_tolerance": 1e-5,
        "passed": True,
    }

    json_path = track_a.output_dir / "track_a_p1.json"
    assert json.loads(json_path.read_text())["evaluations"] == r["evaluations"]

    client = mlflow.MlflowClient(tracking_uri=track_a.tracking_dir.as_uri())
    runs = client.search_runs([client.get_experiment_by_name("avr").experiment_id])
    assert len(runs) == 1
    params = runs[0].data.params
    manifest_digest = hashlib.sha256(track_a.manifest_path.read_bytes()).hexdigest()
    assert params["dataset_manifest_hash"] == manifest_digest
    assert (
        params["split_manifest_hash"] == hashlib.sha256(track_a.split_path.read_bytes()).hexdigest()
    )
    assert params["generation_mapping_hash"] == "not_applicable"
    assert params["encoder_id"] == "toy"
    assert params["encoder_checksum"] == "0" * 64
    assert params["preprocessing_id"] == "toy_pre+p1"
    assert params["prototype_method"] == "m0_global_mean"
    assert params["support_k"] == "full"
    assert params["seed"] == "73"
    assert runs[0].data.metrics["official_full/make_model/top1"] == pytest.approx(2 / 3)


def test_track_a_rerun_is_byte_identical(track_a: GateInputs):
    first = run_track(track_a)
    second = run_track(track_a)

    assert np.array_equal(first.prototypes, second.prototypes)
    for name in first.ranks:
        assert np.array_equal(first.ranks[name], second.ranks[name])
    assert first.results["determinism"] == second.results["determinism"]
    assert first.results["evaluations"] == second.results["evaluations"]


def test_tampered_split_sidecar_is_rejected(track_a: GateInputs):
    track_a.split_path.with_suffix(".sha256").write_text("0" * 64 + "  x\n")
    with pytest.raises(ValueError, match="sha256"):
        run_track(track_a)


# --- Track B ---------------------------------------------------------------------------


@pytest.fixture
def track_b(common: dict) -> GateInputs:
    tmp = common["tmp"]
    # provisional classes: G1 = 1/1/2010, G2 = 1/1/2011 (same model), H = 2/3/2014
    rows = [
        manifest_row("g1s", 1, 1, 2010),
        manifest_row("g1v", 1, 1, 2010),
        manifest_row("g1t", 1, 1, 2010),
        manifest_row("g2s", 1, 1, 2011),
        manifest_row("g2v", 1, 1, 2011),
        manifest_row("g2t", 1, 1, 2011),
        manifest_row("hs", 2, 3, 2014),
        manifest_row("hv", 2, 3, 2014),
        manifest_row("ht", 2, 3, 2014),
        manifest_row("hu", 2, 3, 2014, viewpoint=-1),  # test row, not query-eligible
    ]
    manifest = tmp / "manifest.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=PUBLIC_SCHEMA), manifest)

    def split_row(image_id: str, cls: str, make: int, model: int, part: str, view: int = 1):
        return {
            "image_id": image_id,
            "sha256": image_id * 4,
            "class_id": cls,
            "make_id": make,
            "model_id": model,
            "viewpoint": view,
            "partition": part,
            "mapping_status": "provisional",
        }

    split = pa.Table.from_pylist(
        [
            split_row("g1s", "1/1/2010", 1, 1, "support"),
            split_row("g1v", "1/1/2010", 1, 1, "validation"),
            split_row("g1t", "1/1/2010", 1, 1, "test"),
            split_row("g2s", "1/1/2011", 1, 1, "support"),
            split_row("g2v", "1/1/2011", 1, 1, "validation"),
            split_row("g2t", "1/1/2011", 1, 1, "test"),
            split_row("hs", "2/3/2014", 2, 3, "support"),
            split_row("hv", "2/3/2014", 2, 3, "validation"),
            split_row("ht", "2/3/2014", 2, 3, "test"),
            split_row("hu", "2/3/2014", 2, 3, "test", view=-1),
        ],
        schema=TRACK_B_SCHEMA,
    )
    split_path = write_split_manifest(split, tmp / "track_b" / "seed_73.parquet").manifest
    # support: G1 on x, G2 on y, H on z. test queries: g1t = (0, .9, .1) -> G2 (same model,
    # wrong generation); g2t = y -> G2; ht = z -> H. hu is not query-eligible.
    write_store(
        tmp / "emb",
        {
            "g1s": X,
            "g1v": X,
            "g1t": unit(0.0, 0.9, 0.1),
            "g2s": Y,
            "g2v": Y,
            "g2t": Y,
            "hs": Z,
            "hv": Z,
            "ht": Z,
            "hu": Z,
        },
    )
    return GateInputs(
        track="track_b",
        variant="p1",
        encoder="toy",
        embeddings_dir=tmp / "emb",
        manifest_path=manifest,
        split_path=split_path,
        mapping_queue_path=common["queue"],
        seed=73,
        parity_queries=1000,
        **{k: v for k, v in common.items() if k.endswith("_dir")},
    )


def test_track_b_levels_and_error_decomposition(track_b: GateInputs, common: dict):
    result = run_track(track_b)
    r = result.results

    assert r["mapping_status"] == "provisional"
    assert r["article_result"] is False
    test = r["evaluations"]["test"]
    assert test["support_images"] == 3 and test["query_images"] == 3
    assert test["queries_not_eligible_unknown_viewpoint"] == 1
    assert set(test["levels"]) == {"class", "model", "make"}
    cls = test["levels"]["class"]
    assert cls["classification"]["accuracy"] == pytest.approx(2 / 3)
    # per class F1: G1 0 (no tp), G2 2/3 (p 1/2, r 1), H 1 -> macro F1 5/9
    assert cls["classification"]["macro_f1"] == pytest.approx(5 / 9)
    assert test["levels"]["model"]["classification"]["accuracy"] == 1.0
    assert test["levels"]["model"]["classification"]["class_count"] == 2
    assert test["levels"]["make"]["ranking"]["top_k"]["1"] == 1.0
    d = test["error_decomposition"]
    assert d["wrong_generation_same_model"] == 1
    assert d["generation_accuracy_given_model"] == pytest.approx(2 / 3)
    assert r["evaluations"]["validation"]["levels"]["class"]["ranking"]["top_k"]["1"] == 1.0

    table, _ = read_prototype_table(
        track_b.prototypes_dir / "toy" / "p1" / "m0_track_b_seed_73.parquet"
    )
    assert table.column("class_id").to_pylist() == ["1/1/2010", "1/1/2011", "2/3/2014"]
    assert table.column("year").to_pylist() == [2010, 2011, 2014]
    assert table.column("generation_id").to_pylist() == ["1/1/2010", "1/1/2011", "2/3/2014"]
    assert table.column("mapping_status").to_pylist() == ["provisional"] * 3
    assert r["qdrant"]["collection"].startswith("track_b_seed_73__v0.1_provisional__toy__")
    assert r["qdrant"]["parity"]["queries"] == 3
    assert (
        r["run_manifest"]["generation_mapping_hash"]
        == hashlib.sha256(common["queue"].read_bytes()).hexdigest()
    )


def test_image_listed_as_support_and_query_is_rejected(track_b: GateInputs):
    table = pq.read_table(track_b.split_path).to_pandas()
    table.loc[table.image_id == "g1t", "partition"] = "support"
    doubled = pd.concat([table, table[table.image_id == "g1t"].assign(partition="test")])
    write_split_manifest(
        pa.Table.from_pandas(doubled, schema=TRACK_B_SCHEMA, preserve_index=False),
        track_b.split_path,
    )
    with pytest.raises(ValueError, match="more than once"):
        run_track(track_b)
