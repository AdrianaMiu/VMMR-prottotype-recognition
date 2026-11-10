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
    [
        "data/paths",
        "data/external",
        "data/masks",
        "data/embeddings",
        "data/qdrant",
        "data/qdrant_server",
    ],
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


def _renamed(relative_dir: str, suffix: str):
    """Context manager yielding a dummy file renamed to ``suffix`` inside ``relative_dir``."""

    @contextmanager
    def manager() -> Iterator[Path]:
        with dummy_file(relative_dir) as path:
            target = path.with_name(path.name.replace(".tmp", suffix))
            path.rename(target)
            try:
                yield target
            finally:
                target.rename(path)

    return manager()


def test_the_official_track_a_table_stays_in_git():
    with _renamed("data/splits/track_a", ".parquet") as parquet:
        assert not is_ignored(parquet), "data/splits/track_a/*.parquet is under 2 MB and committed"


@pytest.mark.parametrize(
    "relative_dir",
    [
        "data/manifests",
        "data/splits/track_b",
        "data/splits/track_b/cross_view/seed_1",
        "data/prototypes/toy/p1",
    ],
)
def test_dvc_owns_the_large_parquet_tables(relative_dir: str):
    with _renamed(relative_dir, ".parquet") as parquet:
        assert is_ignored(parquet), f"{relative_dir}/*.parquet belongs to DVC, not git"
    with _renamed(relative_dir, ".parquet.dvc") as pointer:
        assert not is_ignored(pointer), "the .dvc pointer file must be committable"
    with _renamed(relative_dir, ".sha256") as sidecar:
        assert not is_ignored(sidecar), "sha256 sidecars must be committable"


def test_fixture_parquet_and_the_dvc_config_stay_committable():
    with _renamed("tests/fixtures/e0_10class", ".parquet") as parquet:
        assert not is_ignored(parquet)
    assert not is_ignored(REPO_ROOT / ".dvc" / "config")
    assert not is_ignored(REPO_ROOT / ".dvcignore")
    assert is_ignored(REPO_ROOT / ".dvc" / "cache" / "x")
