"""The Track B split manifests must regenerate byte-identically (PROTOCOL.md §5 Track B).

Split manifests are stored, never regenerated implicitly: the published ``.sha256``
sidecars under ``data/splits/track_b/`` pin every seed. Where the image manifest is
present (it is built from the CompCars release and is not committed), each seed is
rebuilt into a temporary directory and compared with its sidecar; the sidecar and
summary checks run everywhere.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from avr.config import CONFIG_DIR, load_config
from avr.data.manifest import MANIFEST_NAME
from avr.data.split_track_b import EligibilityThresholds, build_track_b_splits
from avr.io import read_sidecar_sha256

REPO_ROOT = CONFIG_DIR.parent
MANIFEST = REPO_ROOT / "data" / "manifests" / f"{MANIFEST_NAME}.parquet"
SPLITS = REPO_ROOT / "data" / "splits" / "track_b"
CFG = load_config().split_track_b
SEEDS = [CFG.primary_seed, *CFG.sensitivity_seeds]


def published_summary() -> dict:
    return json.loads((SPLITS / "summary.json").read_text())


@pytest.mark.skipif(not MANIFEST.is_file(), reason="image manifest not built on this machine")
def test_rebuilt_splits_match_the_published_sidecars(tmp_path: Path):
    manifest_digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    assert manifest_digest == read_sidecar_sha256(MANIFEST.with_suffix(".sha256"))

    result = build_track_b_splits(
        pd.read_parquet(MANIFEST),
        EligibilityThresholds(**{k: CFG[k] for k in EligibilityThresholds.model_fields}),
        dict(CFG.fractions),
        seeds=SEEDS,
        audit_path=tmp_path / "audit.md",
        out_dir=tmp_path / "splits",
    )

    for seed in SEEDS:
        rebuilt = hashlib.sha256(result.outputs[seed].manifest.read_bytes()).hexdigest()
        assert rebuilt == read_sidecar_sha256(SPLITS / f"seed_{seed}.sha256"), seed
    assert result.partition_counts == {
        int(s): c for s, c in published_summary()["partition_counts"].items()
    }


def test_published_sidecars_cover_the_primary_and_sensitivity_seeds():
    assert SEEDS == [73, 13, 37, 101, 137]
    for seed in SEEDS:
        assert len(read_sidecar_sha256(SPLITS / f"seed_{seed}.sha256")) == 64
    summary = published_summary()
    assert summary["manifest_sha256"] == read_sidecar_sha256(MANIFEST.with_suffix(".sha256"))
    for seed in SEEDS:
        counts = summary["partition_counts"][str(seed)]
        assert counts["support"] > counts["validation"] >= counts["test"] > 0
