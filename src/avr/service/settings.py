"""Service settings from the environment (prefix ``AVR_``), PROTOCOL.md §16 behaviour knobs.

Paths default to the repository layout (``configs/service/profiles.yaml``, ``data/prototypes``,
``data/qdrant``); a deployment points them elsewhere with ``AVR_PROTOTYPES_DIR`` and friends.
``return_embeddings`` stays disabled unless ``AVR_ALLOW_EMBEDDINGS=true`` (a development
flag, never set in a public deployment) and ``POST /v1/indexes/reload`` is disabled until
``AVR_RELOAD_TOKEN`` is set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from avr.config import CONFIG_DIR
from avr.encoders.registry import ENCODERS_DIR
from avr.service.profiles import PROFILES_PATH

REPO_ROOT = CONFIG_DIR.parent
SearchBackendName = Literal["exact", "qdrant_embedded"]


class ServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AVR_", extra="ignore", frozen=True)

    profiles_path: Path = PROFILES_PATH
    prototypes_dir: Path = REPO_ROOT / "data" / "prototypes"
    qdrant_dir: Path = REPO_ROOT / "data" / "qdrant"
    encoders_dir: Path = ENCODERS_DIR
    search_backend: SearchBackendName = "qdrant_embedded"
    device: str | None = None
    reload_token: str | None = None
    allow_embeddings: bool = False
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    max_image_pixels: int = Field(default=40_000_000, gt=0)
    max_top_k: int = Field(default=100, ge=1)
    default_top_k: int = Field(default=10, ge=1)
