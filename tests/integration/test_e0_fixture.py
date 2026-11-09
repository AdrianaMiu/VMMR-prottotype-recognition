"""E0 on the committed ten-class fixture (PROTOCOL.md §14 "Stage E0", issue M1-15).

Runs in CI with no dataset and no GPU: manifest validation, M0 prototypes, exact search,
ranking and classification metrics and the run manifest, replayed from
``tests/fixtures/e0_10class/`` and compared with the metric values recorded when the fixture
was built. Any change to those steps that alters a metric fails here. torch is never
imported, which the subprocess test checks in a fresh interpreter.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from avr.pipeline.e0 import EXPECTED_NAME, run_e0

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "e0_10class"
TOLERANCE = 1e-9


def expected_metrics() -> dict[str, float]:
    return json.loads((FIXTURE_DIR / EXPECTED_NAME).read_text())


def test_every_metric_equals_the_recorded_value(tmp_path):
    expected = expected_metrics()

    metrics = run_e0(FIXTURE_DIR, tracking_dir=tmp_path / "mlruns")

    assert set(metrics) == set(expected)
    for name, value in expected.items():
        assert abs(metrics[name] - value) <= TOLERANCE, (name, metrics[name], value)


def test_rerun_is_deterministic(tmp_path):
    first = run_e0(FIXTURE_DIR, tracking_dir=tmp_path / "first")
    second = run_e0(FIXTURE_DIR, tracking_dir=tmp_path / "second")

    assert first == second


def test_pipeline_runs_without_importing_torch(tmp_path):
    program = (
        "import json, sys\n"
        "from avr.pipeline.e0 import run_e0\n"
        f"metrics = run_e0({str(FIXTURE_DIR)!r}, tracking_dir={str(tmp_path / 'mlruns')!r})\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
        "print(json.dumps(metrics))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=True
    )
    metrics = json.loads(result.stdout.strip().splitlines()[-1])
    assert metrics["query_count"] == expected_metrics()["query_count"]
