"""Run manifest (PROTOCOL.md §17).

The manifest is the record that makes a metric citable (GLOSSARY.md: "no manifest, no
metric"). Its fields are exactly the §17 list, in that order. Every field may be ``None``
while a run is being assembled; :meth:`RunManifest.is_complete` is what the tracking layer
checks before it accepts a metric, and only ``end_time`` is allowed to be missing at that
point because it is written when the run closes.
"""

from __future__ import annotations

import platform
import subprocess
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path

from pydantic import BaseModel, ConfigDict

_OPTIONAL_AT_METRIC_TIME = ("end_time",)


class RunManifest(BaseModel):
    """Exactly the PROTOCOL.md §17 fields, in protocol order."""

    model_config = ConfigDict(extra="forbid")

    run_id: str | None = None
    git_commit: str | None = None
    dirty_worktree_flag: bool | None = None
    dataset_protocol: str | None = None
    dataset_manifest_hash: str | None = None
    split_manifest_hash: str | None = None
    generation_mapping_hash: str | None = None
    encoder_id: str | None = None
    encoder_revision: str | None = None
    encoder_checksum: str | None = None
    preprocessing_id: str | None = None
    dimension_transform_id: str | None = None
    prototype_method: str | None = None
    support_k: int | None = None
    seed: int | None = None
    hardware: dict[str, str | None] | None = None
    software_versions: dict[str, str] | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None

    def missing_fields(self) -> list[str]:
        """Required fields (everything but ``end_time``) that are still ``None``."""
        return [name for name in REQUIRED_FIELDS if getattr(self, name) is None]

    def is_complete(self) -> bool:
        return not self.missing_fields()


REQUIRED_FIELDS: tuple[str, ...] = tuple(
    name for name in RunManifest.model_fields if name not in _OPTIONAL_AT_METRIC_TIME
)


# --- environment capture helpers -------------------------------------------------------


def _git(args: list[str], cwd: Path | None) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def capture_git_commit(repo: Path | None = None) -> str:
    """Full SHA of ``HEAD`` in ``repo`` (default: current directory)."""
    return _git(["rev-parse", "HEAD"], repo)


def capture_dirty_worktree_flag(repo: Path | None = None) -> bool:
    """``True`` when ``git status --porcelain`` reports any tracked or untracked change."""
    return bool(_git(["status", "--porcelain"], repo))


def _cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


def _gpu_name() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    names = [line.strip() for line in out.splitlines() if line.strip()]
    return ", ".join(names) or None


def capture_hardware() -> dict[str, str | None]:
    """CPU model and GPU name (``None`` when ``nvidia-smi`` is absent or fails)."""
    return {"cpu": _cpu_model(), "gpu": _gpu_name(), "platform": platform.platform()}


def capture_software_versions() -> dict[str, str]:
    """Interpreter and core library versions; ``torch`` only when importable."""
    import numpy
    import pandas

    import avr

    versions = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "avr": avr.__version__,
    }
    try:
        versions["torch"] = import_module("torch").__version__
    except ImportError:
        pass
    return versions


def capture_environment(repo: Path | None = None) -> dict:
    """The manifest fields that describe where and when the run started."""
    return {
        "git_commit": capture_git_commit(repo),
        "dirty_worktree_flag": capture_dirty_worktree_flag(repo),
        "hardware": capture_hardware(),
        "software_versions": capture_software_versions(),
        "start_time": datetime.now(UTC),
    }
