"""The publication policy in docs/protocols/compcars-licence.md relies on .gitignore.

These tests create real dummy files inside the repository checkout and ask git itself
whether they are ignored, so they prove the rules rather than re-reading .gitignore.
"""

import subprocess
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

REPO_ROOT = Path(
    subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path(__file__).parent,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
)


@contextmanager
def dummy_file(relative_dir: str) -> Iterator[Path]:
    """Create a uniquely named file under ``relative_dir`` and remove it afterwards."""
    directory = REPO_ROOT / relative_dir
    created_dirs = []
    for parent in reversed([directory, *directory.parents]):
        if not parent.exists():
            parent.mkdir()
            created_dirs.append(parent)
    path = directory / f"gitignore-policy-{uuid.uuid4().hex}.tmp"
    path.write_text("dummy\n")
    try:
        yield path
    finally:
        path.unlink()
        for parent in reversed(created_dirs):
            parent.rmdir()


def is_ignored(path: Path) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", str(path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    return result.returncode == 0


@pytest.mark.parametrize(
    "relative_dir",
    ["data/paths", "data/external", "data/masks", "data/embeddings", "data/qdrant"],
)
def test_path_table_and_external_data_are_ignored(relative_dir: str):
    with dummy_file(relative_dir) as path:
        assert is_ignored(path), f"{relative_dir}/ must be git-ignored"
    assert not path.exists()


def test_public_manifest_json_is_not_ignored():
    with dummy_file("data/manifests") as path:
        json_path = path.with_suffix(".json")
        path.rename(json_path)
        try:
            assert not is_ignored(json_path), "data/manifests/*.json must be committable"
        finally:
            json_path.rename(path)
    assert not path.exists()


def test_dedup_pair_panels_are_ignored_but_the_report_is_not():
    with dummy_file("reports/dedup/pairs") as panel:
        assert is_ignored(panel), "reports/dedup/pairs/ holds CompCars pixels"
    with dummy_file("reports/dedup") as note:
        assert not is_ignored(note), "reports/dedup/ text must be committable"


def test_public_split_manifests_are_not_ignored():
    with dummy_file("data/splits/track_a") as path:
        parquet = path.with_suffix(".parquet")
        path.rename(parquet)
        try:
            assert not is_ignored(parquet), "data/splits/**/*.parquet must be committable"
        finally:
            parquet.rename(path)
    assert not path.exists()


def test_prototype_parquet_is_ignored_but_its_sidecar_is_not():
    with dummy_file("data/prototypes/toy/p1") as path:
        parquet = path.with_suffix(".parquet")
        sidecar = path.with_suffix(".sha256")
        path.rename(parquet)
        try:
            assert is_ignored(parquet), "data/prototypes/**/*.parquet must be git-ignored"
            parquet.rename(sidecar)
            assert not is_ignored(sidecar), "prototype .sha256 sidecars must be committable"
        finally:
            (sidecar if sidecar.exists() else parquet).rename(path)
    assert not path.exists()
