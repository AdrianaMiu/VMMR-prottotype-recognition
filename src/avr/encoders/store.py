"""Resumable per-shard embedding store (PROTOCOL.md §9 step 6, ADR-0002).

Layout: ``<root>/<encoder>/<variant>/shard_<k>.parquet`` with columns ``image_id`` (string)
and ``embedding`` (fixed-size list of float32, already L2-normalised), plus ``shard_<k>.json``
holding the :class:`ShardManifest`: the encoder identity (name, revision, checkpoint
checksum, preprocessing id, variant, compute dtype) and the SHA-256 of the Parquet file. A
shard counts as finished only when both files exist, and the JSON is written after the
Parquet, so an interrupted run never leaves a half shard that a rerun would trust. ``<root>``
is ``data/embeddings/`` by default and git-ignored.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from avr.config import CONFIG_DIR
from avr.io import read_fixed_list_matrix, sha256_of_file

DEFAULT_ROOT = CONFIG_DIR.parent / "data" / "embeddings"
_SHARD_RE = re.compile(r"^shard_(\d{5})\.parquet$")


class StoreIdentity(BaseModel):
    """What produced the embeddings; every shard of one store carries the same identity."""

    model_config = ConfigDict(frozen=True)

    encoder: str
    revision: str
    checkpoint_sha256: str
    preprocessing_id: str
    variant: str
    dtype: str


class ShardManifest(StoreIdentity):
    shard_index: int
    row_count: int
    dim: int
    shard_sha256: str
    written_at: datetime


def embedding_table(image_ids: Sequence[str], embeddings: np.ndarray) -> pa.Table:
    """The shard table: ``image_id`` strings and ``embedding`` fixed-size float32 lists."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    dim = int(embeddings.shape[1])
    return pa.table(
        {
            "image_id": pa.array(list(image_ids), pa.string()),
            "embedding": pa.FixedSizeListArray.from_arrays(
                pa.array(embeddings.reshape(-1), pa.float32()), dim
            ),
        }
    )


def read_embedding_table(path: Path) -> tuple[list[str], np.ndarray]:
    """Read a table written by :func:`embedding_table` as ``(image_ids, [n, dim] float32)``."""
    table = pq.read_table(path)
    return table.column("image_id").to_pylist(), read_fixed_list_matrix(table, "embedding")


class EmbeddingStore:
    def __init__(self, root: Path, identity: StoreIdentity) -> None:
        self.root = Path(root)
        self.identity = identity
        self.directory = self.root / identity.encoder / identity.variant

    @classmethod
    def from_directory(cls, root: Path, encoder: str, variant: str) -> EmbeddingStore:
        """Open an existing store, taking its identity from the first finished shard."""
        directory = Path(root) / encoder / variant
        manifests = sorted(directory.glob("shard_*.json"))
        if not manifests:
            raise FileNotFoundError(f"{directory}: no shard manifest found")
        shard = ShardManifest.model_validate_json(manifests[0].read_text())
        identity = StoreIdentity.model_validate(shard.model_dump())
        if identity.encoder != encoder or identity.variant != variant:
            raise ValueError(f"{directory}: shard identity {identity} does not match the path")
        return cls(root, identity)

    # --- paths ---------------------------------------------------------------------------

    def shard_path(self, index: int) -> Path:
        return self.directory / f"shard_{index:05d}.parquet"

    def shard_manifest_path(self, index: int) -> Path:
        return self.directory / f"shard_{index:05d}.json"

    def has_shard(self, index: int) -> bool:
        return self.shard_path(index).exists() and self.shard_manifest_path(index).exists()

    def finished_shards(self) -> list[int]:
        if not self.directory.exists():
            return []
        indices = []
        for path in self.directory.iterdir():
            match = _SHARD_RE.match(path.name)
            if match and self.has_shard(int(match.group(1))):
                indices.append(int(match.group(1)))
        return sorted(indices)

    # --- writing -------------------------------------------------------------------------

    def _check_identity(self) -> None:
        for index in self.finished_shards():
            existing = StoreIdentity.model_validate(self.read_manifest(index).model_dump())
            if existing != self.identity:
                differing = [
                    k
                    for k in StoreIdentity.model_fields
                    if getattr(existing, k) != getattr(self.identity, k)
                ]
                raise ValueError(
                    f"{self.directory} already holds shards with a different identity "
                    f"(fields: {', '.join(differing)})"
                )
            break  # every finished shard shares one identity, checking one is enough

    def write_shard(
        self, index: int, image_ids: list[str], embeddings: np.ndarray
    ) -> ShardManifest:
        embeddings = np.asarray(embeddings)
        if embeddings.dtype != np.float32:
            raise ValueError(f"embeddings must be float32, got {embeddings.dtype}")
        if embeddings.ndim != 2 or embeddings.shape[0] != len(image_ids):
            raise ValueError(
                f"embeddings shape {embeddings.shape} does not match {len(image_ids)} image ids"
            )
        self._check_identity()
        dim = int(embeddings.shape[1])
        table = embedding_table(image_ids, embeddings)
        self.directory.mkdir(parents=True, exist_ok=True)
        parquet = self.shard_path(index)
        tmp = parquet.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp, compression="zstd")
        tmp.replace(parquet)
        manifest = ShardManifest(
            **self.identity.model_dump(),
            shard_index=index,
            row_count=len(image_ids),
            dim=dim,
            shard_sha256=sha256_of_file(parquet),
            written_at=datetime.now(UTC).replace(microsecond=0),
        )
        manifest_tmp = self.shard_manifest_path(index).with_suffix(".json.tmp")
        manifest_tmp.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n")
        manifest_tmp.replace(self.shard_manifest_path(index))
        return manifest

    # --- reading -------------------------------------------------------------------------

    def read_manifest(self, index: int) -> ShardManifest:
        return ShardManifest.model_validate_json(self.shard_manifest_path(index).read_text())

    def load_shard(self, index: int) -> tuple[list[str], np.ndarray]:
        return read_embedding_table(self.shard_path(index))

    def load_all(self) -> tuple[list[str], np.ndarray]:
        """Every finished shard in index order as ``(image_ids, [n, dim] float32)``."""
        indices = self.finished_shards()
        if not indices:
            return [], np.zeros((0, 0), dtype=np.float32)
        ids: list[str] = []
        blocks = []
        for index in indices:
            shard_ids, block = self.load_shard(index)
            ids.extend(shard_ids)
            blocks.append(block)
        return ids, np.vstack(blocks)
