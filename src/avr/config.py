"""Hydra configuration entry point.

All experiment choices live under ``configs/`` (PROTOCOL.md §17: configuration files,
never hard-coded choices). ``load_config`` composes that tree without touching the
process-wide Hydra state, so it is safe to call from tests and scripts alike.
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_config(
    config_name: str = "config",
    overrides: list[str] | None = None,
    config_dir: Path = CONFIG_DIR,
) -> DictConfig:
    """Compose the configuration tree rooted at ``config_dir``.

    ``overrides`` use Hydra syntax, e.g. ``["encoder=dinov3_vitb16"]``.
    """
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose(config_name=config_name, overrides=list(overrides or []))
