"""Loading the prototype Parquet tables into a Qdrant target (issue M5-1).

The toy tables are written with :func:`write_prototype_table` into the same
``<encoder>/<variant>/m0_<protocol_id>.parquet`` layout the M1 gate uses; the expected
collection names are spelled out by hand from PROTOCOL.md §12.
"""

import numpy as np
import pandas as pd
import pytest

from avr.prototypes.persist import PrototypeProvenance, prototype_table, write_prototype_table
from avr.retrieval.qdrant_load import load_prototype_tables
from avr.retrieval.qdrant_store import LocalTarget, list_collections, query_top_k


def toy_table(track: str, encoder: str, variant: str):
    class_ids = ["c0", "c1", "c2"]
    status = "not_applicable" if track == "track_a" else "provisional"
    payload = pd.DataFrame(
        {
            "class_id": class_ids,
            "make_id": [1, 1, 2],
            "make_name": ["A", "A", "B"],
            "model_id": [10, 11, 20],
            "model_name": ["a", "b", "c"],
            "year": [None, None, None] if track == "track_a" else [2010, 2011, 2012],
            "generation_id": [None] * 3 if track == "track_a" else ["g0", "g1", "g2"],
            "mapping_status": [status] * 3,
        }
    )
    provenance = PrototypeProvenance(
        prototype_method="m0_global_mean",
        encoder=encoder,
        encoder_revision="r1",
        encoder_checksum="0" * 64,
        preprocessing_id=f"toy_pre+{variant}",
        run_id=f"gate_{track}",
    )
    return prototype_table(np.eye(3, dtype=np.float32), class_ids, [2, 3, 4], payload, provenance)


@pytest.fixture
def prototypes_dir(tmp_path):
    root = tmp_path / "prototypes"
    write_prototype_table(
        toy_table("track_a", "toy", "p1"), root / "toy" / "p1" / "m0_track_a_official.parquet"
    )
    write_prototype_table(
        toy_table("track_b", "toy", "p3"), root / "toy" / "p3" / "m0_track_b_seed_7.parquet"
    )
    return root


def test_every_table_becomes_a_collection_named_by_protocol_section_12(tmp_path, prototypes_dir):
    target = LocalTarget(path=tmp_path / "qdrant")

    loaded = load_prototype_tables(prototypes_dir, target)

    expected = [
        "track_a_official__none__toy__toy_pre+p1__3__m0_global_mean",
        "track_b_seed_7__v0.1_provisional__toy__toy_pre+p3__3__m0_global_mean",
    ]
    assert [summary.collection for summary in loaded] == expected
    assert [summary.point_count for summary in loaded] == [3, 3]
    assert list_collections(target) == expected
    hits = query_top_k(target, expected[1], np.eye(3, dtype=np.float32)[1:2], k=1)
    assert hits[0][0].class_id == "c1"
    assert hits[0][0].payload["generation_id"] == "g1"


def test_a_table_whose_sidecar_disagrees_is_refused(tmp_path, prototypes_dir):
    sidecar = prototypes_dir / "toy" / "p1" / "m0_track_a_official.sha256"
    sidecar.write_text("0" * 64 + "  m0_track_a_official.parquet\n")

    with pytest.raises(ValueError, match="sha256"):
        load_prototype_tables(prototypes_dir, LocalTarget(path=tmp_path / "qdrant"))


def test_an_empty_directory_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_prototype_tables(tmp_path / "nothing", LocalTarget(path=tmp_path / "qdrant"))
