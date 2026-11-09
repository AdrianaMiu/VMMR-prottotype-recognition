"""Five-prototype fixture shared by the service tests (same vectors as test_qdrant_export)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from avr.prototypes.persist import PrototypeProvenance, prototype_table, write_prototype_table
from avr.retrieval.qdrant_store import export_prototypes

PROVENANCE = PrototypeProvenance(
    prototype_method="m0_global_mean",
    encoder="toy",
    encoder_revision="r1",
    encoder_checksum="0" * 64,
    preprocessing_id="toy_pre+p1",
    run_id="gate_toy",
)
FIVE_CLASS_IDS = [f"c{k}" for k in range(5)]
FIVE_COLLECTION = "toy__none__toy__toy_pre+p1__5__m0_global_mean"


@pytest.fixture
def five_prototypes():
    """Unit vectors along the five axes of R^5; class k lives on axis k."""
    payload = pd.DataFrame(
        {
            "class_id": FIVE_CLASS_IDS,
            "make_id": [1, 1, 2, 2, 3],
            "make_name": ["A", "A", "B", "B", "C"],
            "model_id": [10, 11, 20, 21, 30],
            "model_name": ["a", "b", "c", "d", "e"],
            "year": [2010, 2011, None, 2013, 2014],
            "generation_id": ["g0", "g1", None, "g3", "g4"],
            "mapping_status": ["provisional"] * 5,
        }
    )
    return prototype_table(
        np.eye(5, dtype=np.float32), FIVE_CLASS_IDS, [2] * 5, payload, PROVENANCE
    )


@pytest.fixture
def five_prototype_files(tmp_path: Path, five_prototypes) -> dict[str, Path]:
    """The fixture table on disk (Parquet + sidecar) and exported to an embedded Qdrant store."""
    parquet = tmp_path / "prototypes" / "toy" / "p1" / "m0_toy.parquet"
    write_prototype_table(five_prototypes, parquet)
    qdrant = tmp_path / "qdrant"
    export_prototypes(qdrant, FIVE_COLLECTION, five_prototypes)
    return {"parquet": parquet, "qdrant": qdrant}
