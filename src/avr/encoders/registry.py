"""Encoder registry schema and loader (PROTOCOL.md §9, ADR-0003).

One YAML file under ``configs/encoders/`` describes one frozen encoder checkpoint: the
identity fields §9 requires (model id, source, revision, licence, parameter count, input
resolution, native dimension, pooling, preprocessing, dtype, checkpoint checksum) plus
the record of the verification run that measured the pooled output shape for real.
``native_dim`` is never inferred from transformer width: once an entry is verified it
must equal the last element of ``verified_output_shape``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from avr.config import CONFIG_DIR

EncoderSource = Literal["huggingface", "open_clip"]
Pooling = Literal["cls", "attention_pool", "mean_patch", "projection"]
TorchDtype = Literal["float16", "bfloat16", "float32"]

NAME_PATTERN = r"^[a-z0-9]+(?:_[a-z0-9]+)*$"
SHA256_PATTERN = r"^[0-9a-f]{64}$"

ENCODERS_DIR = CONFIG_DIR / "encoders"


class PreprocessingSpec(BaseModel):
    """The checkpoint's official preprocessing, named by a stable id.

    Checkpoint-specific settings (e.g. ``max_num_patches`` for NaFlex) are kept as extra
    keys so the registry records exactly what the processor was configured with.
    """

    model_config = ConfigDict(extra="allow")

    id: str = Field(pattern=NAME_PATTERN)


class EncoderSpec(BaseModel):
    """One registry entry; field order is the YAML key order written by the verifier."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    name: str = Field(pattern=NAME_PATTERN)
    model_id: str
    source: EncoderSource
    revision: str
    license: str
    parameter_count: int | None = Field(default=None, gt=0)
    input_resolution: int = Field(gt=0)
    native_dim: int = Field(gt=0)
    pooling: Pooling
    preprocessing: PreprocessingSpec
    dtype: TorchDtype
    checkpoint_file: str
    checkpoint_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    verified_at: datetime | None = None
    verified_dtype: TorchDtype | None = None
    verified_output_shape: list[int] | None = None

    @model_validator(mode="after")
    def _native_dim_matches_verified_shape(self) -> EncoderSpec:
        shape = self.verified_output_shape
        if shape is not None and (not shape or shape[-1] != self.native_dim):
            raise ValueError(
                f"native_dim {self.native_dim} does not equal the last element of "
                f"verified_output_shape {shape}"
            )
        return self


def load_registry(encoders_dir: Path = ENCODERS_DIR) -> dict[str, EncoderSpec]:
    """Validate every ``*.yaml`` under ``encoders_dir`` and return the entries by name."""
    registry: dict[str, EncoderSpec] = {}
    for path in sorted(encoders_dir.glob("*.yaml")):
        spec = EncoderSpec.model_validate(yaml.safe_load(path.read_text()))
        if spec.name != path.stem:
            raise ValueError(f"{path.name}: entry name {spec.name!r} must match the file stem")
        if spec.name in registry:
            raise ValueError(f"duplicate encoder name {spec.name!r}")
        registry[spec.name] = spec
    return registry


def write_entry(path: Path, spec: EncoderSpec) -> None:
    """Write ``spec`` to ``path`` in schema key order, keeping any leading ``#`` comment lines."""
    comment_lines: list[str] = []
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.startswith("#"):
                break
            comment_lines.append(line)
    body = yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    header = "".join(f"{line}\n" for line in comment_lines)
    path.write_text(header + body)
