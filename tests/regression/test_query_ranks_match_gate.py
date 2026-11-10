"""Per-query tables must reproduce the committed ``reports/m1_gate/track_*_p*.json`` exactly.

Top-1, top-5, MRR and mean rank recomputed from the per-query table equal the committed gate
numbers of the primary evaluation (Track A ``official_full``, Track B ``test``). Runs only
where the public image manifest, the split tables and the finished DINOv3 ViT-B/16 embedding
store are present (``AVR_EMBEDDINGS_DIR`` overrides ``data/embeddings``; ``AVR_MANIFEST`` and
``AVR_SPLITS_DIR`` override the manifest and ``data/splits``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from avr.config import CONFIG_DIR
from avr.pipeline.query_ranks import QueryRanksInputs, run_query_ranks

REPO_ROOT = CONFIG_DIR.parent
ENCODER = "dinov3_vitb16"
EMBEDDINGS_DIR = Path(os.environ.get("AVR_EMBEDDINGS_DIR", REPO_ROOT / "data" / "embeddings"))
MANIFEST = Path(
    os.environ.get("AVR_MANIFEST", REPO_ROOT / "data" / "manifests" / "compcars_web_images.parquet")
)
SPLITS_DIR = Path(os.environ.get("AVR_SPLITS_DIR", REPO_ROOT / "data" / "splits"))
SPLITS = {
    "track_a": SPLITS_DIR / "track_a" / "official_cvpr2015.parquet",
    "track_b": SPLITS_DIR / "track_b" / "seed_73.parquet",
}
PRIMARY = {"track_a": "official_full", "track_b": "test"}


@pytest.mark.parametrize("variant", ["p1", "p3"])
@pytest.mark.parametrize("track", ["track_a", "track_b"])
@pytest.mark.skipif(
    not (MANIFEST.exists() and (EMBEDDINGS_DIR / ENCODER).exists() and SPLITS["track_a"].exists()),
    reason="manifest, splits or embedding store not available",
)
def test_per_query_aggregates_equal_committed_gate_numbers(track, variant):
    gate = json.loads((REPO_ROOT / "reports" / "m1_gate" / f"{track}_{variant}.json").read_text())
    gate = gate["evaluations"][PRIMARY[track]]
    level = gate["levels"]["make_model" if track == "track_a" else "class"]["ranking"]
    result = run_query_ranks(
        QueryRanksInputs(
            track=track,
            variant=variant,
            encoder=ENCODER,
            embeddings_dir=EMBEDDINGS_DIR,
            manifest_path=MANIFEST,
            split_path=SPLITS[track],
        )
    )
    table = result.per_query
    assert len(table) == gate["query_images"] == level["query_count"]
    assert result.prototype_count == gate["class_count"]
    assert table.hit_at_1.mean() == level["top_k"]["1"]
    assert table.hit_at_5.mean() == level["top_k"]["5"]
    assert table.hit_at_10.mean() == level["top_k"]["10"]
    assert table.hit_at_20.mean() == level["top_k"]["20"]
    assert (1.0 / table.rank_true).mean() == level["mrr"]
    assert table.rank_true.mean() == level["mean_rank"]
    assert table.rank_true.median() == level["median_rank"]
    assert int(table.tied.sum()) >= level["tied_queries"]
    if track == "track_a":
        assert set(table.split_label) == {gate_split_label()}
        assert set(table.mapping_status) == {"not_applicable"}
    else:
        assert set(table.mapping_status) == {"provisional"}


def gate_split_label() -> str:
    from avr.data.split_identity import TRACK_A_LABEL

    return TRACK_A_LABEL
