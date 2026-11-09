"""Nothing committed under reports/ or data/**/*.json may carry a machine-specific path.

Every path written by the renderers (M1 gate JSON and README, leakage audits) is relative
to the repository root (``avr.io.repo_relative``), so a ``/home/`` prefix is a defect. The
scan covers every tracked or committable file git sees under ``reports/`` and ``data/``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SUFFIXES = {".md", ".json"}


def committable_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "reports", "data"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    return sorted(REPO_ROOT / p for p in out if Path(p).suffix in SUFFIXES)


FILES = committable_files()


def test_the_scan_covers_the_gate_readme_and_result_files():
    names = {p.relative_to(REPO_ROOT).as_posix() for p in FILES}
    assert "reports/m1_gate/README.md" in names
    assert "reports/m1_gate/track_a_p1.json" in names
    assert "reports/leakage/e0_fixture.md" in names


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.relative_to(REPO_ROOT).as_posix())
def test_no_machine_specific_path_is_committed(path: Path):
    assert "/home/" not in path.read_text(errors="replace"), path
