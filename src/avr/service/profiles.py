"""Service profiles (PROTOCOL.md §16; configs/service/profiles.yaml).

A profile is the service-side name of one experiment configuration: track, encoder,
preprocessing variant, dimension transform and prototype method. Loading a profile resolves
the encoder entry in the registry (``avr.encoders.registry``), builds the §12 collection
name through ``avr.retrieval.qdrant_store.collection_name`` and reads the prototype table's
SHA-256, which is what the service reports as ``index_version``. Loading fails closed: a
missing table or a table whose bytes no longer match its ``.sha256`` sidecar is an error,
never a silently different index.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from avr.config import CONFIG_DIR
from avr.encoders.registry import ENCODERS_DIR, EncoderSpec, load_registry
from avr.preprocessing.variants import Variant
from avr.retrieval.qdrant_store import collection_name

PROFILES_PATH = CONFIG_DIR / "service" / "profiles.yaml"
PROFILE_ID_PATTERN = r"^[a-z0-9]+(?:_[a-z0-9]+)*(?:__[a-z0-9]+(?:_[a-z0-9]+)*){4}$"
Track = Literal["track_a", "track_b"]


class ProfileSpec(BaseModel):
    """One entry of ``profiles.yaml`` as written."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str = Field(pattern=PROFILE_ID_PATTERN)
    track: Track
    dataset_protocol: str
    mapping_version: str
    encoder: str
    variant: Variant
    dimension: str = "native"
    prototype_method: str
    prototype_table: str
    accept_threshold: float = Field(default=0.6, ge=-1.0, le=1.0)


class ProfilesFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_profile_id: str
    profiles: list[ProfileSpec] = Field(min_length=1)


class Profile(BaseModel):
    """A resolved profile: the spec plus everything derived from the registry and the table."""

    model_config = ConfigDict(frozen=True)

    spec: ProfileSpec
    encoder_spec: EncoderSpec
    prototype_path: Path
    index_version: str
    prototype_count: int

    @property
    def profile_id(self) -> str:
        return self.spec.profile_id

    @property
    def preprocessing_id(self) -> str:
        return f"{self.encoder_spec.preprocessing.id}+{self.spec.variant}"

    @property
    def dimension_value(self) -> int:
        if self.spec.dimension == "native":
            return self.encoder_spec.native_dim
        return int(self.spec.dimension)

    @property
    def collection(self) -> str:
        return collection_name(
            self.spec.dataset_protocol,
            self.spec.mapping_version,
            self.spec.encoder,
            self.preprocessing_id,
            self.dimension_value,
            self.spec.prototype_method,
        )


class ProfileRegistry:
    def __init__(self, default_profile_id: str, profiles: list[Profile]) -> None:
        self._profiles = {p.profile_id: p for p in profiles}
        if default_profile_id not in self._profiles:
            raise ValueError(f"default_profile_id {default_profile_id!r} is not a profile")
        self.default_profile_id = default_profile_id

    def __getitem__(self, profile_id: str) -> Profile:
        return self._profiles[profile_id]

    def __contains__(self, profile_id: object) -> bool:
        return profile_id in self._profiles

    def __iter__(self) -> Iterator[Profile]:
        return iter(self._profiles.values())

    def __len__(self) -> int:
        return len(self._profiles)

    @property
    def default(self) -> Profile:
        return self._profiles[self.default_profile_id]


def _checked_sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"prototype table {path} does not exist")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = path.with_suffix(".sha256")
    if sidecar.exists():
        recorded = sidecar.read_text().split()[0]
        if recorded != digest:
            raise ValueError(f"{path}: sha256 {digest} differs from the sidecar {sidecar.name}")
    return digest


def _row_count(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def load_profiles(
    path: Path = PROFILES_PATH,
    prototypes_dir: Path = Path("data/prototypes"),
    encoders_dir: Path = ENCODERS_DIR,
) -> ProfileRegistry:
    """Validate ``profiles.yaml`` and resolve every profile against the registry and the tables."""
    try:
        doc = ProfilesFile.model_validate(yaml.safe_load(Path(path).read_text()))
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    encoders = load_registry(Path(encoders_dir))
    profiles = []
    for spec in doc.profiles:
        if spec.encoder not in encoders:
            raise ValueError(f"{spec.profile_id}: encoder {spec.encoder!r} is not in the registry")
        table_path = Path(prototypes_dir) / spec.prototype_table
        profiles.append(
            Profile(
                spec=spec,
                encoder_spec=encoders[spec.encoder],
                prototype_path=table_path,
                index_version=_checked_sha256(table_path),
                prototype_count=_row_count(table_path),
            )
        )
    ids = [p.profile_id for p in profiles]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate profile_id in profiles.yaml")
    return ProfileRegistry(doc.default_profile_id, profiles)
