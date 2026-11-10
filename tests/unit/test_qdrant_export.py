"""Qdrant export of the prototype table in embedded local mode (PROTOCOL.md §12, M1-16).

Five fake prototypes; the expected neighbours are read off the vectors by hand. Embedded
mode needs no server, no GPU extra and no network.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from avr.prototypes.persist import PrototypeProvenance, prototype_table
from avr.retrieval.qdrant_store import (
    KEYWORD_FIELDS,
    collection_name,
    export_prototypes,
    query_top_k,
)

PROVENANCE = PrototypeProvenance(
    prototype_method="m0_global_mean",
    encoder="toy",
    encoder_revision="r1",
    encoder_checksum="0" * 64,
    preprocessing_id="toy_pre+p1",
    run_id="gate_toy",
)


@pytest.fixture
def five_prototypes():
    # unit vectors along the five axes of R^5; class k lives on axis k
    class_ids = [f"c{k}" for k in range(5)]
    payload = pd.DataFrame(
        {
            "class_id": class_ids,
            "make_id": [1, 1, 2, 2, 3],
            "make_name": ["A", "A", "B", "B", "C"],
            "model_id": [10, 11, 20, 21, 30],
            "model_name": ["a", "b", "c", "d", "e"],
            "year": [2010, 2011, None, 2013, 2014],
            "generation_id": ["g0", "g1", None, "g3", "g4"],
            "mapping_status": ["provisional"] * 5,
        }
    )
    return prototype_table(np.eye(5, dtype=np.float32), class_ids, [2] * 5, payload, PROVENANCE)


def test_collection_name_follows_protocol_section_12():
    assert (
        collection_name(
            dataset_protocol="track_b_seed_73",
            mapping_version="v0.1_provisional",
            encoder="dinov3_vitb16",
            preprocessing="dinov3_vit16_224_v1+p1",
            dimension=768,
            prototype_method="m0_global_mean",
        )
        == "track_b_seed_73__v0.1_provisional__dinov3_vitb16__dinov3_vit16_224_v1+p1__768"
        "__m0_global_mean"
    )


def test_export_writes_every_prototype_with_keyword_payload(tmp_path, five_prototypes):
    summary = export_prototypes(tmp_path / "qdrant", "toy__coll", five_prototypes)

    assert summary.point_count == 5
    assert summary.dimension == 5
    assert summary.collection == "toy__coll"
    assert tuple(summary.keyword_indexes) == KEYWORD_FIELDS
    assert KEYWORD_FIELDS == (
        "make_id",
        "make_name",
        "model_id",
        "model_name",
        "year",
        "generation_id",
        "mapping_status",
    )
    assert summary.payload_sample["make_id"] == "1"
    assert summary.payload_sample["class_id"] == "c0"
    assert (tmp_path / "qdrant").is_dir()


def test_exact_top_k_matches_hand_ranking_and_null_payload_fields_are_kept(
    tmp_path, five_prototypes
):
    export_prototypes(tmp_path / "qdrant", "toy__coll", five_prototypes)
    # query (0.9, 0.3, 0.1, 0, 0): nearest c0, then c1, then c2
    queries = np.array([[0.9, 0.3, 0.1, 0.0, 0.0], [0.0, 0.0, 0.0, 0.1, 0.8]], dtype=np.float32)

    hits = query_top_k(tmp_path / "qdrant", "toy__coll", queries, k=3)

    assert [h.class_id for h in hits[0]] == ["c0", "c1", "c2"]
    assert [h.class_id for h in hits[1]][:2] == ["c4", "c3"]
    assert hits[0][2].payload["year"] is None
    assert hits[0][0].payload["make_name"] == "A"


def test_export_replaces_an_existing_collection(tmp_path, five_prototypes):
    export_prototypes(tmp_path / "qdrant", "toy__coll", five_prototypes)
    summary = export_prototypes(tmp_path / "qdrant", "toy__coll", five_prototypes.slice(0, 3))
    assert summary.point_count == 3
    hits = query_top_k(tmp_path / "qdrant", "toy__coll", np.eye(5, dtype=np.float32)[4:5], k=5)
    assert len(hits[0]) == 3


def test_target_union_parses_path_and_url_forms():
    from avr.retrieval.qdrant_store import LocalTarget, ServerTarget, resolve_target

    local = resolve_target({"path": "data/qdrant"})
    assert isinstance(local, LocalTarget)
    assert local.path == Path("data/qdrant")

    server = resolve_target({"url": "http://localhost:6333"})
    assert isinstance(server, ServerTarget)
    assert server.url == "http://localhost:6333"
    assert server.api_key is None
    assert server.timeout_s == 30

    with_key = resolve_target({"url": "http://q:6333", "api_key": "s3cret", "timeout_s": 5})
    assert with_key.api_key == "s3cret"
    assert with_key.timeout_s == 5

    assert resolve_target(Path("x")) == LocalTarget(path=Path("x"))
    assert resolve_target(local) is local
    with pytest.raises(ValueError):
        resolve_target({"path": "a", "url": "http://b"})
    with pytest.raises(ValueError):
        resolve_target({})


def test_local_target_exports_and_queries_like_a_path(tmp_path, five_prototypes):
    from avr.retrieval.qdrant_store import LocalTarget, list_collections

    target = LocalTarget(path=tmp_path / "qdrant")
    export_prototypes(target, "toy__coll", five_prototypes)
    hits = query_top_k(target, "toy__coll", np.eye(5, dtype=np.float32)[2:3], k=1, exact=True)
    assert [h.class_id for h in hits[0]] == ["c2"]
    assert list_collections(target) == ["toy__coll"]
