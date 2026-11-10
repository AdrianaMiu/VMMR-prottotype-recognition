"""``build_fixture`` end to end on a toy release: two classes, three images each, a colour
embedder (PROTOCOL.md §14 "Stage E0").

The embedder maps a crop to its L2-normalised mean RGB, so the embedding stored for an
image can be checked against the colour the image was painted with, independently of the
code under test.
"""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from avr.data.manifest import PRIVATE_SCHEMA, PUBLIC_SCHEMA
from avr.encoders.store import StoreIdentity, read_embedding_table
from avr.pipeline.e0 import (
    EMBEDDINGS_NAME,
    EXPECTED_NAME,
    README_NAME,
    SLICE_NAME,
    SPLIT_NAME,
    build_fixture,
)

IDENTITY = StoreIdentity(
    encoder="colour",
    revision="r1",
    checkpoint_sha256="1" * 64,
    preprocessing_id="colour_pre",
    variant="p1",
    dtype="float32",
)
COLOURS = {
    "a": (200, 10, 10),
    "b": (10, 200, 10),
}


class ColourEmbedder:
    native_dim = 3

    def __init__(self) -> None:
        self.batches: list[int] = []

    def embed(self, images):
        self.batches.append(len(images))
        out = np.stack([np.asarray(im, dtype=np.float32).mean(axis=(0, 1)) for im in images])
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def manifest_row(image_id: str, make: int, model: int) -> dict:
    return {
        "image_id": image_id,
        "sha256": image_id * 4,
        "width": 20,
        "height": 20,
        "make_id": make,
        "make_name": f"make{make}",
        "model_id": model,
        "model_name": f"model{model}",
        "year": 2010,
        "viewpoint": 1,
        "bbox_xyxy": [2, 2, 17, 17],
        "duplicate_group_id": image_id,
        "source_group_id": image_id,
        "validity_status": "valid",
    }


@pytest.fixture
def toy_release(tmp_path):
    root = tmp_path / "release"
    ids = [f"{c}{i}" for c in "ab" for i in range(3)]
    rows, paths = [], []
    for image_id in ids:
        make = 1 if image_id[0] == "a" else 2
        rows.append(manifest_row(image_id, make, make))
        rel = f"image/{image_id}.png"
        paths.append({"image_id": image_id, "relative_path": rel})
        (root / "image").mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 20), COLOURS[image_id[0]]).save(root / rel)
    manifest = tmp_path / "manifest.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=PUBLIC_SCHEMA), manifest)
    path_table = tmp_path / "paths.parquet"
    pq.write_table(pa.Table.from_pylist(paths, schema=PRIVATE_SCHEMA), path_table)
    return root, manifest, path_table


def test_build_fixture_writes_every_file_and_the_colour_embeddings(toy_release, tmp_path):
    root, manifest, path_table = toy_release
    embedder = ColourEmbedder()
    fixture_dir = tmp_path / "fixture"

    metrics = build_fixture(
        fixture_dir,
        manifest_path=manifest,
        paths_path=path_table,
        root=root,
        embedder=embedder,
        identity=IDENTITY,
        git_commit="deadbeef",
        n_classes=2,
        n_support=2,
        n_query=1,
        seed=73,
        batch_size=4,
        expand=0.05,
        tracking_dir=tmp_path / "mlruns",
    )

    for name in (SLICE_NAME, EMBEDDINGS_NAME, SPLIT_NAME, EXPECTED_NAME, README_NAME):
        assert (fixture_dir / name).exists(), name
    assert embedder.batches == [4, 2]

    split = json.loads((fixture_dir / SPLIT_NAME).read_text())
    assert split["class_ids"] == ["001_0001", "002_0002"]
    assert split["seed"] == 73
    assert split["embedding_identity"]["encoder"] == "colour"
    assert len(split["source_manifest_sha256"]) == 64

    slice_table = pq.read_table(fixture_dir / SLICE_NAME)
    assert slice_table.num_rows == 6
    assert "relative_path" not in slice_table.schema.names

    ids, matrix = read_embedding_table(fixture_dir / EMBEDDINGS_NAME)
    assert len(ids) == 6
    red = np.array([200, 10, 10], dtype=np.float32)
    red /= np.linalg.norm(red)
    for image_id, vector in zip(ids, matrix, strict=True):
        if image_id.startswith("a"):
            assert np.allclose(vector, red, atol=1e-6), image_id

    expected = json.loads((fixture_dir / EXPECTED_NAME).read_text())
    assert expected == metrics
    assert expected["query_count"] == 2
    assert expected["top1"] == 1.0

    readme = (fixture_dir / README_NAME).read_text()
    assert "not the Track A" in readme
    assert "colour" in readme
