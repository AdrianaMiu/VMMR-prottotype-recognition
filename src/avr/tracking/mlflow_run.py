"""MLflow wiring for the run manifest (PROTOCOL.md §17, ADR-0002: local file store).

:func:`tracked_run` opens one MLflow run on a local file store, logs every manifest field as
a parameter, writes the manifest as ``manifest.json`` and hands back a :class:`TrackedRun`.
``TrackedRun.log_metric`` enforces "no manifest, no metric": it raises
:class:`IncompleteManifestError` while any required field is still ``None``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import mlflow

from avr.tracking.manifest import RunManifest

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRACKING_DIR = REPO_ROOT / "mlruns"
DEFAULT_EXPERIMENT = "avr"
MANIFEST_ARTIFACT = "manifest.json"


class IncompleteManifestError(RuntimeError):
    """Raised when a metric is logged against a manifest with missing required fields."""


@dataclass
class TrackedRun:
    manifest: RunManifest
    mlflow_run_id: str

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        missing = self.manifest.missing_fields()
        if missing:
            raise IncompleteManifestError(
                f"refusing to log metric {name!r}: manifest is missing {', '.join(missing)}"
            )
        mlflow.log_metric(name, value, step=step, run_id=self.mlflow_run_id)

    def log_param(self, name: str, value: str) -> None:
        """An input outside the §17 list (for example a mask-metadata digest)."""
        mlflow.MlflowClient().log_param(self.mlflow_run_id, name, value)


def _manifest_json(manifest: RunManifest) -> str:
    return json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=False)


def _write_manifest_artifact(manifest: RunManifest, run_id: str) -> None:
    mlflow.log_text(_manifest_json(manifest), MANIFEST_ARTIFACT, run_id=run_id)


@contextmanager
def tracked_run(
    manifest: RunManifest,
    tracking_dir: Path | str = DEFAULT_TRACKING_DIR,
    experiment_name: str = DEFAULT_EXPERIMENT,
) -> Iterator[TrackedRun]:
    """Open an MLflow run on the file store at ``tracking_dir`` for ``manifest``.

    Parameters are logged on entry (``None`` fields as the string ``"None"``), the manifest
    artifact is written on entry and rewritten on exit once ``end_time`` is set. The run is
    marked FAILED if the body raises.
    """
    tracking_dir = Path(tracking_dir)
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
    mlflow.set_experiment(experiment_name)
    if manifest.start_time is None:
        manifest.start_time = datetime.now(UTC)

    with mlflow.start_run(run_name=manifest.run_id) as active:
        run_id = active.info.run_id
        mlflow.log_params(
            {k: str(v) for k, v in manifest.model_dump(mode="json").items()}, run_id=run_id
        )
        _write_manifest_artifact(manifest, run_id)
        try:
            yield TrackedRun(manifest=manifest, mlflow_run_id=run_id)
        finally:
            manifest.end_time = datetime.now(UTC)
            _write_manifest_artifact(manifest, run_id)


def tracking_dir_from_config(tracking_cfg) -> Path:
    """Resolve ``cfg.tracking.tracking_dir`` (configs/tracking/*.yaml) against the repo root."""
    path = Path(str(tracking_cfg.tracking_dir))
    return path if path.is_absolute() else REPO_ROOT / path
