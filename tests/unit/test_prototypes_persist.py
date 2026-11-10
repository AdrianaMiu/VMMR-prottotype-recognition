"""Persisted M0 prototype table (issue M1-16 addendum; PROTOCOL.md §12 payload fields)."""

import hashlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from avr.prototypes.persist import (
    PROTOTYPE_COLUMNS,
    PrototypeProvenance,
    prototype_table,
    write_prototype_table,
)

PROVENANCE = PrototypeProvenance(
    prototype_method="m0_global_mean",
    encoder="toy",
    encoder_revision="r1",
    encoder_checksum="0" * 64,
    preprocessing_id="toy_pre+p1",
    run_id="gate_toy",
)


def payload_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "class_id": ["b", "a"],
            "make_id": [2, 1],
            "make_name": ["MakeB", "MakeA"],
            "model_id": [20, 10],
            "model_name": ["ModelB", "ModelA"],
            "year": [2012, None],
            "generation_id": ["2/20/2012", None],
            "mapping_status": ["provisional", "not_applicable"],
        }
    )


def test_table_has_the_addendum_columns_in_class_id_order():
    prototypes = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    table = prototype_table(prototypes, ["b", "a"], [3, 5], payload_frame(), PROVENANCE)

    assert table.schema.names == list(PROTOTYPE_COLUMNS)
    assert PROTOTYPE_COLUMNS == (
        "class_id",
        "embedding",
        "support_count",
        "make_id",
        "make_name",
        "model_id",
        "model_name",
        "year",
        "generation_id",
        "mapping_status",
        "prototype_method",
        "encoder",
        "encoder_revision",
        "encoder_checksum",
        "preprocessing_id",
        "dimension",
        "run_id",
    )
    rows = table.to_pylist()
    assert [r["class_id"] for r in rows] == ["a", "b"]
    assert rows[0]["embedding"] == [1.0, 0.0]
    assert rows[0]["support_count"] == 5
    assert rows[0]["year"] is None and rows[0]["generation_id"] is None
    assert rows[1]["year"] == 2012 and rows[1]["mapping_status"] == "provisional"
    assert rows[0]["dimension"] == 2 and rows[0]["encoder_checksum"] == "0" * 64
    assert rows[0]["run_id"] == "gate_toy"


def test_payload_class_missing_or_duplicated_is_rejected():
    prototypes = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="payload"):
        prototype_table(prototypes, ["b", "c"], [1, 1], payload_frame(), PROVENANCE)
    doubled = pd.concat([payload_frame(), payload_frame()])
    with pytest.raises(ValueError, match="payload"):
        prototype_table(prototypes, ["b", "a"], [1, 1], doubled, PROVENANCE)


def test_write_is_byte_identical_and_sidecar_matches(tmp_path):
    prototypes = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    table = prototype_table(prototypes, ["b", "a"], [3, 5], payload_frame(), PROVENANCE)

    first = write_prototype_table(table, tmp_path / "one" / "m0_x.parquet")
    second = write_prototype_table(table, tmp_path / "two" / "m0_x.parquet")

    digest = hashlib.sha256(first.parquet.read_bytes()).hexdigest()
    assert first.sha256.read_text() == f"{digest}  m0_x.parquet\n"
    assert first.parquet.read_bytes() == second.parquet.read_bytes()
    assert pq.read_table(first.parquet).equals(table)
