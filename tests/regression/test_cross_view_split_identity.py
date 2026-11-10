"""The Track B cross-view variants must regenerate byte-identically (ADR-0004).

The published ``.sha256`` sidecars under ``data/splits/track_b/cross_view/seed_<s>/`` pin
every held-out view of every seed. Where the parent split tables are present (they are
larger than the commit limit and not committed), each variant is rebuilt into a temporary
directory and compared with its sidecar; the sidecar and summary checks run everywhere.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from avr.config import CONFIG_DIR, load_config
from avr.data.cross_view import CrossViewThresholds, build_cross_view_splits
from avr.io import read_sidecar_sha256

REPO_ROOT = CONFIG_DIR.parent
SPLITS = REPO_ROOT / "data" / "splits" / "track_b"
CROSS_VIEW = SPLITS / "cross_view"
CFG = load_config().split_track_b
SEEDS = [CFG.primary_seed, *CFG.sensitivity_seeds]
VIEWS = list(CFG.cross_view.views)
PARENTS_PRESENT = all((SPLITS / f"seed_{s}.parquet").is_file() for s in SEEDS)


def published_summary() -> dict:
    return json.loads((CROSS_VIEW / "summary.json").read_text())


@pytest.mark.skipif(not PARENTS_PRESENT, reason="parent split tables not built on this machine")
def test_rebuilt_variants_match_the_published_sidecars(tmp_path: Path):
    summary = published_summary()
    for seed in SEEDS:
        parent = SPLITS / f"seed_{seed}.parquet"
        digest = hashlib.sha256(parent.read_bytes()).hexdigest()
        assert digest == read_sidecar_sha256(parent.with_suffix(".sha256")), seed
        assert digest == summary["parent_sha256"][str(seed)], seed

        result = build_cross_view_splits(
            pd.read_parquet(parent),
            seed=seed,
            views=VIEWS,
            thresholds=CrossViewThresholds(
                min_support=CFG.cross_view.min_support, min_test=CFG.cross_view.min_test
            ),
            report_path=tmp_path / f"seed_{seed}.md",
            out_dir=tmp_path / f"seed_{seed}",
        )
        for view in VIEWS:
            rebuilt = hashlib.sha256(result.outputs[view].manifest.read_bytes()).hexdigest()
            published = CROSS_VIEW / f"seed_{seed}" / f"view_{view}.sha256"
            assert rebuilt == read_sidecar_sha256(published), (seed, view)
        assert result.class_counts == {
            int(v): n for v, n in summary["class_counts"][str(seed)].items()
        }
        assert result.partition_counts == {
            int(v): c for v, c in summary["partition_counts"][str(seed)].items()
        }


def test_published_sidecars_cover_every_seed_and_held_out_view():
    assert SEEDS == [73, 13, 37, 101, 137]
    assert VIEWS == [1, 2, 3, 4, 5]
    summary = published_summary()
    for seed in SEEDS:
        assert summary["parent_sha256"][str(seed)] == read_sidecar_sha256(
            SPLITS / f"seed_{seed}.sha256"
        )
        for view in VIEWS:
            sidecar = CROSS_VIEW / f"seed_{seed}" / f"view_{view}.sha256"
            assert len(read_sidecar_sha256(sidecar)) == 64, (seed, view)
            counts = summary["partition_counts"][str(seed)][str(view)]
            assert counts["support"] > counts["test"] > 0
            assert counts["validation"] > 0
            assert 0 < summary["class_counts"][str(seed)][str(view)] <= 2645
