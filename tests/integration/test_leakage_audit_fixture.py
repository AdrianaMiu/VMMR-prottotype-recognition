"""Leakage audit on the committed E0 fixture (PROTOCOL.md §23 Data; issue M1-12).

Runs in CI with no dataset and no GPU. The fixture README states 10 classes with 5
support and 3 query images each and no duplicate group spanning support and query; the
counts asserted here are those statements, not values read back from the audit.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from avr.data.leakage import e0_fixture_findings

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "e0_10class"
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_leakage_audit.py"


def test_fixture_passes_every_must_pass_check():
    findings, inputs = e0_fixture_findings(FIXTURE_DIR)
    by_check = {f.check: f for f in findings}

    assert all(f.passed for f in findings), [f.check for f in findings if not f.passed]
    assert by_check["split_ids_resolve_in_manifest"].counts["split_rows"] == 80
    assert by_check["no_duplicate_group_id_crosses_partition"].counts == {
        "groups": 80,
        "crossing_groups": 0,
        "crossing_images": 0,
    }
    assert by_check["every_query_class_has_support"].counts == {
        "target_classes": 10,
        "classes_without_support": 0,
    }
    assert by_check["fixture_slice_checksum"].passed
    assert inputs["manifest_slice sha256"].startswith("f3daae1176b15413")


def test_cli_audits_the_fixture_and_writes_the_report(tmp_path: Path):
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--fixture",
            str(FIXTURE_DIR),
            "--reports-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = (tmp_path / "e0_fixture.md").read_text()
    assert "Verdict: **PASS**" in report
    assert "split_ids_resolve_in_manifest" in report


def test_cli_exits_non_zero_when_a_duplicate_is_planted(tmp_path: Path):
    import json
    import shutil

    planted = tmp_path / "fixture"
    shutil.copytree(FIXTURE_DIR, planted)
    split = json.loads((planted / "split.json").read_text())
    first = split["class_ids"][0]
    # list one support image of the first class as a query too: its duplicate group now
    # sits on both sides
    split["query"][first].append(split["support"][first][-1])
    (planted / "split.json").write_text(json.dumps(split))

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--fixture", str(planted), "--reports-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "no_duplicate_group_id_crosses_partition" in proc.stdout + proc.stderr
