import pytest

from avr.tracking.manifest import RunManifest
from avr.tracking.mlflow_run import IncompleteManifestError, tracked_run


def test_logging_a_metric_without_a_complete_manifest_raises(tmp_path):
    manifest = RunManifest(run_id="r1", seed=73)

    with tracked_run(manifest, tracking_dir=tmp_path / "mlruns") as run:
        with pytest.raises(IncompleteManifestError, match="git_commit"):
            run.log_metric("top1", 0.5)


def test_tracking_dir_from_config_resolves_relative_paths_against_repo_root():
    from avr.config import load_config
    from avr.tracking.mlflow_run import REPO_ROOT, tracking_dir_from_config

    cfg = load_config()

    assert tracking_dir_from_config(cfg.tracking) == REPO_ROOT / "mlruns"
    assert (REPO_ROOT / "pyproject.toml").exists()
