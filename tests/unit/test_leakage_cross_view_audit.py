"""``run_leakage_audit.py --cross-view`` on a synthetic parent split with a planted violation.

The parent seed table and its view variants are written into a temporary repository layout
(``data/splits/track_b/...``), the script's module-level paths are pointed at it, and the
expected findings are read off the hand-built rows.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from avr.data.cross_view import CrossViewThresholds, cross_view_variant
from avr.data.manifest import PUBLIC_SCHEMA
from avr.data.split_track_b import OUTPUT_SCHEMA, split_table
from avr.data.splits_io import write_split_manifest
from avr.io import write_sha256_sidecar

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_leakage_audit.py"
VIEWS = [1, 2]


def load_script():
    spec = importlib.util.spec_from_file_location("run_leakage_audit_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def manifest_rows() -> list[dict]:
    rows = []
    for i in range(14):
        rows.append(
            {
                "image_id": f"i{i:02d}",
                "sha256": f"h{i:02d}",
                "make_id": 1,
                "model_id": 1,
                "model_name": "m",
                "make_name": "k",
                "year": 2010,
                "viewpoint": 1 if i < 7 else 2,
                "bbox_xyxy": [0, 0, 5, 5],
                "duplicate_group_id": f"i{i:02d}",
                "source_group_id": f"i{i:02d}",
                "validity_status": "valid",
            }
        )
    return rows


def parent_table() -> pd.DataFrame:
    """One class: view 1 holds 5 support + 1 test + 1 validation, view 2 the same."""
    spec = []
    for base in (0, 7):
        spec += [(f"i{base + k:02d}", "support") for k in range(5)]
        spec += [(f"i{base + 5:02d}", "test"), (f"i{base + 6:02d}", "validation")]
    frame = pd.DataFrame(spec, columns=["image_id", "partition"])
    frame["sha256"] = frame.image_id.map(lambda i: "h" + i[1:])
    frame["class_id"] = "1/1/2010"
    frame["make_id"] = 1
    frame["model_id"] = 1
    frame["viewpoint"] = [1] * 7 + [2] * 7
    frame["mapping_status"] = "provisional"
    return frame


@pytest.fixture
def repo(tmp_path: Path, monkeypatch) -> tuple:
    module = load_script()
    manifest = tmp_path / "data" / "manifests" / "compcars_web_images.parquet"
    manifest.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(manifest_rows(), schema=PUBLIC_SCHEMA), manifest)
    write_sha256_sidecar(manifest)
    splits = tmp_path / "data" / "splits" / "track_b"
    write_split_manifest(split_table(parent_table()), splits / "seed_73.parquet")
    parent = pd.read_parquet(splits / "seed_73.parquet")  # as build_cross_view_splits.py does
    thresholds = CrossViewThresholds(min_support=5, min_test=1)
    for view in VIEWS:
        variant = cross_view_variant(parent, view, thresholds)
        write_split_manifest(
            split_table(variant), splits / "cross_view" / "seed_73" / f"view_{view}.parquet"
        )
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "MANIFEST", manifest)
    monkeypatch.setattr(module, "TRACK_B_DIR", splits)
    monkeypatch.setattr(module, "CROSS_VIEW_DIR", splits / "cross_view")
    monkeypatch.setattr(module, "CROSS_VIEW_SEEDS", [73])
    monkeypatch.setattr(module, "CROSS_VIEW_VIEWS", VIEWS)
    monkeypatch.setattr(module, "CROSS_VIEW_THRESHOLDS", thresholds)
    return module, tmp_path, splits


def test_clean_cross_view_variants_pass_and_the_report_is_written(repo):
    module, tmp_path, _ = repo
    reports = tmp_path / "reports" / "leakage"

    (result,) = module.audit_cross_view(reports, tmp_path / "scratch")

    assert result.split == "track_b_cross_view_seed_73"
    assert result.passed, [(f.check, f.counts) for f in result.findings if not f.passed]
    text = (reports / "track_b_cross_view_seed_73.md").read_text()
    assert "Verdict: **PASS**" in text
    for view in VIEWS:
        assert f"view_{view}/partition_matches_parent_seed" in text
        assert f"view_{view}/regeneration_byte_identical" in text
    assert "/home/" not in text and str(tmp_path) not in text


def test_a_planted_partition_change_fails_the_audit(repo):
    module, tmp_path, splits = repo
    path = splits / "cross_view" / "seed_73" / "view_1.parquet"
    table = pq.read_table(path).to_pandas()
    moved = table.index[table.partition == "support"][0]
    table.loc[moved, "partition"] = "test"  # re-assigned instead of filtered
    write_split_manifest(
        pa.Table.from_pandas(table, schema=OUTPUT_SCHEMA, preserve_index=False), path
    )

    (result,) = module.audit_cross_view(tmp_path / "reports", tmp_path / "scratch")

    assert not result.passed
    assert "view_1/partition_matches_parent_seed" in result.failed_checks
    assert "view_1/regeneration_byte_identical" in result.failed_checks


def test_a_tampered_parent_sidecar_stops_the_audit(repo):
    module, tmp_path, splits = repo
    (splits / "seed_73.sha256").write_text("0" * 64 + "  seed_73.parquet\n")
    with pytest.raises(SystemExit, match="sidecar"):
        module.audit_cross_view(tmp_path / "reports", tmp_path / "scratch")
