"""Resumable per-shard embedding store (PROTOCOL.md §9 step 6, ADR-0002)."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from avr.encoders.store import EmbeddingStore, ShardManifest, StoreIdentity

IDENTITY = StoreIdentity(
    encoder="fake_encoder",
    revision="0123456789abcdef0123456789abcdef01234567",
    checkpoint_sha256="a" * 64,
    preprocessing_id="fake_pre_v1",
    variant="p1",
    dtype="float16",
)


def _vectors(seed: int, n: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@pytest.fixture
def store(tmp_path: Path) -> EmbeddingStore:
    return EmbeddingStore(tmp_path / "embeddings", IDENTITY)


def test_store_directory_is_encoder_then_variant(tmp_path: Path, store: EmbeddingStore):
    assert store.directory == tmp_path / "embeddings" / "fake_encoder" / "p1"


def test_round_trip_of_random_vectors_is_exact(store: EmbeddingStore):
    ids = [f"{i:016x}" for i in range(5)]
    vectors = _vectors(1, 5)

    store.write_shard(0, ids, vectors)
    loaded_ids, loaded = store.load_shard(0)

    assert loaded_ids == ids
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, vectors)


def test_shard_parquet_has_image_id_and_fixed_size_embedding_columns(store: EmbeddingStore):
    store.write_shard(0, ["a", "b"], _vectors(2, 2))
    schema = pq.read_schema(store.shard_path(0))
    assert schema.names == ["image_id", "embedding"]
    kind = schema.field("embedding").type
    assert pa.types.is_fixed_size_list(kind)
    assert kind.list_size == 8 and kind.value_type == pa.float32()


def test_shard_manifest_records_identity_and_the_parquet_sha256(store: EmbeddingStore):
    store.write_shard(0, ["a", "b", "c"], _vectors(3, 3))

    manifest = json.loads(store.shard_manifest_path(0).read_text())
    expected_sha = hashlib.sha256(store.shard_path(0).read_bytes()).hexdigest()
    assert manifest["encoder"] == "fake_encoder"
    assert manifest["revision"] == IDENTITY.revision
    assert manifest["checkpoint_sha256"] == "a" * 64
    assert manifest["preprocessing_id"] == "fake_pre_v1"
    assert manifest["variant"] == "p1"
    assert manifest["dtype"] == "float16"
    assert manifest["shard_sha256"] == expected_sha
    assert manifest["row_count"] == 3
    assert manifest["dim"] == 8
    assert ShardManifest.model_validate(manifest).shard_index == 0


def test_finished_shards_need_both_parquet_and_manifest(store: EmbeddingStore):
    store.write_shard(0, ["a"], _vectors(4, 1))
    store.write_shard(2, ["b"], _vectors(5, 1))
    store.shard_manifest_path(2).unlink()  # a crash between the two writes

    assert store.finished_shards() == [0]
    assert store.has_shard(0) and not store.has_shard(2)


def test_load_all_concatenates_shards_in_index_order(store: EmbeddingStore):
    v0, v1 = _vectors(6, 3), _vectors(7, 2)
    store.write_shard(1, ["d", "e"], v1)
    store.write_shard(0, ["a", "b", "c"], v0)

    ids, matrix = store.load_all()

    assert ids == ["a", "b", "c", "d", "e"]
    np.testing.assert_array_equal(matrix, np.vstack([v0, v1]))


def test_load_all_on_an_empty_store_is_empty(store: EmbeddingStore):
    ids, matrix = store.load_all()
    assert ids == [] and matrix.shape == (0, 0)


def test_write_rejects_mismatched_lengths_and_non_float32(store: EmbeddingStore):
    with pytest.raises(ValueError):
        store.write_shard(0, ["a"], _vectors(8, 2))
    with pytest.raises(ValueError):
        store.write_shard(0, ["a"], _vectors(8, 1).astype(np.float64))


def test_identity_mismatch_with_an_existing_store_is_rejected(tmp_path: Path):
    EmbeddingStore(tmp_path, IDENTITY).write_shard(0, ["a"], _vectors(9, 1))
    other = IDENTITY.model_copy(update={"checkpoint_sha256": "b" * 64})
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        EmbeddingStore(tmp_path, other).write_shard(1, ["b"], _vectors(10, 1))
