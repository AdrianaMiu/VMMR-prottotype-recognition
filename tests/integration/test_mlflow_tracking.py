import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import mlflow

from avr.tracking.manifest import RunManifest, capture_environment
from avr.tracking.mlflow_run import tracked_run

REPO = Path(__file__).resolve().parents[2]


def complete_manifest() -> RunManifest:
    return RunManifest(
        run_id="dummy-run",
        dataset_protocol="compcars_web",
        dataset_manifest_hash="0" * 64,
        split_manifest_hash="1" * 64,
        generation_mapping_hash="2" * 64,
        encoder_id="dinov3_vitb16",
        encoder_revision="main",
        encoder_checksum="3" * 64,
        preprocessing_id="p1_bbox_crop",
        dimension_transform_id="none",
        prototype_method="m0_global_mean",
        support_k=5,
        seed=73,
        **capture_environment(REPO),
    )


def test_dummy_run_leaves_params_and_manifest_artifact(tmp_path):
    tracking_dir = tmp_path / "mlruns"
    head = subprocess.run(
        ["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()

    with tracked_run(complete_manifest(), tracking_dir=tracking_dir) as run:
        run.log_metric("top1", 0.5)
        mlflow_run_id = run.mlflow_run_id

    client = mlflow.MlflowClient(tracking_uri=tracking_dir.as_uri())
    stored = client.get_run(mlflow_run_id)
    assert stored.data.params["git_commit"] == head
    assert stored.data.params["run_id"] == "dummy-run"
    assert stored.data.params["support_k"] == "5"
    assert stored.data.metrics["top1"] == 0.5
    assert stored.info.status == "FINISHED"

    artifacts = {a.path for a in client.list_artifacts(mlflow_run_id)}
    assert "manifest.json" in artifacts
    (tmp_path / "dl").mkdir()
    local = client.download_artifacts(mlflow_run_id, "manifest.json", str(tmp_path / "dl"))
    manifest = json.loads(Path(local).read_text())
    assert manifest["git_commit"] == head
    assert manifest["end_time"] is not None
    assert datetime.fromisoformat(manifest["end_time"]) <= datetime.now(UTC)
    assert list(manifest) == list(RunManifest.model_fields)
