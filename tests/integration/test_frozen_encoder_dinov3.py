"""Embedding contract (PROTOCOL.md §9, §23 "Embeddings") on the real DINOv3 ViT-B/16.

Needs the ``gpu`` extra, a CUDA device and the cached checkpoint, so it skips otherwise.
Expected dimension (768) is the protocol's value for ViT-B/16, not read from the model.

Tolerance: the encoder runs in fp16 on CUDA. Batched and single-image forward passes take
different CUDA kernels (GEMM tiling depends on the batch dimension), so fp16 activations can
differ by a few units in the last place; after float32 casting and L2 normalisation the
embeddings still differ slightly. Measured on this machine (RTX 3080, torch 2.7, eight
random 400x300 images): max-abs difference 2.3e-4, norm deviation 6e-8. The bound is 1e-3,
about four times the measured difference, so a kernel-level change is tolerated but a wrong
pooling or a missing normalisation (differences of order 1e-1 or more) is not.

The same contract is repeated on the P3 variant (masked crops on the neutral background)
for real CompCars images and the committed full-run masks. That part also needs the public
manifest, the private path table, the CompCars root and the mask store (``AVR_MASKS_DIR``,
default ``data/masks``; the masks are git-ignored and live in the checkout that built them).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from avr.config import CONFIG_DIR, load_config
from avr.encoders.extract import load_jobs, select_p3_jobs
from avr.encoders.registry import ENCODERS_DIR, load_registry
from avr.preprocessing.p3 import load_mask, mask_path
from avr.preprocessing.variants import prepare

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

EXPECTED_DIM = 768
NORM_TOL = 1e-3
BATCH_TOL = 1e-3


def _images(n: int) -> list[Image.Image]:
    rng = np.random.default_rng(73)
    out = []
    for _ in range(n):
        arr = rng.integers(0, 256, size=(300, 400, 3), dtype=np.uint8)
        out.append(Image.fromarray(arr, "RGB"))
    return out


@pytest.fixture(scope="module")
def encoder():
    from avr.encoders.adapter import FrozenEncoder

    spec = load_registry(ENCODERS_DIR)["dinov3_vitb16"]
    return FrozenEncoder.from_spec(spec, device="cuda")


def test_embeddings_have_the_protocol_shape_and_are_finite(encoder):
    out = encoder.embed(_images(3))
    assert out.shape == (3, EXPECTED_DIM)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert encoder.native_dim == EXPECTED_DIM


def test_embeddings_are_unit_norm(encoder):
    out = encoder.embed(_images(4))
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=NORM_TOL)


def test_batch_of_eight_matches_single_image_outputs(encoder):
    images = _images(8)
    batched = encoder.embed(images)
    singles = np.vstack([encoder.embed([im]) for im in images])
    assert np.abs(batched - singles).max() < BATCH_TOL


def test_identity_matches_the_registry_entry(encoder):
    spec = load_registry(ENCODERS_DIR)["dinov3_vitb16"]
    identity = encoder.identity("p1")
    assert identity.encoder == "dinov3_vitb16"
    assert identity.revision == spec.revision
    assert identity.checkpoint_sha256 == spec.checkpoint_sha256
    assert identity.preprocessing_id == "dinov3_vit16_224_v1"
    assert identity.dtype == "float16"
    assert identity.variant == "p1"


def test_save_then_load_is_exact(encoder, tmp_path):
    from avr.encoders.store import EmbeddingStore

    images = _images(5)
    ids = [f"{i:016x}" for i in range(5)]
    out = encoder.embed(images)
    store = EmbeddingStore(tmp_path, encoder.identity("p1"))
    store.write_shard(0, ids, out)

    loaded_ids, loaded = store.load_all()

    assert loaded_ids == ids
    np.testing.assert_array_equal(loaded, out)
    manifest = store.read_manifest(0)
    assert manifest.revision == encoder.revision
    assert manifest.checkpoint_sha256 == encoder.checkpoint_sha256


# --- the same contract on the P3 variant -------------------------------------------------

REPO_ROOT = CONFIG_DIR.parent
MANIFEST = REPO_ROOT / "data" / "manifests" / "compcars_web_images.parquet"
PATHS = REPO_ROOT / "data" / "paths" / "compcars_web_image_paths.parquet"
MASK_METADATA = REPO_ROOT / "data" / "manifests" / "p3_mask_metadata.parquet"
MASKS_DIR = Path(os.environ.get("AVR_MASKS_DIR", REPO_ROOT / "data" / "masks"))
P3_IMAGES = 8


@pytest.fixture(scope="module")
def p3_crops() -> tuple[list[str], list[Image.Image], list[Image.Image]]:
    """Image ids with their P1 and P3 crops for the first ``ok`` images of the manifest."""
    root = (REPO_ROOT / load_config().dataset.root).resolve()
    needed = [MANIFEST, PATHS, MASK_METADATA, MASKS_DIR, root / "image"]
    if not all(p.exists() for p in needed):
        pytest.skip("needs the manifest, path table, mask metadata, mask store and CompCars root")
    jobs = load_jobs(MANIFEST, PATHS, limit=200)
    kept, _ = select_p3_jobs(jobs, MASK_METADATA, MASKS_DIR)
    kept = kept[:P3_IMAGES]
    assert len(kept) == P3_IMAGES
    p1, p3 = [], []
    for job in kept:
        with Image.open(root / job.relative_path) as image:
            rgb = image.convert("RGB")
        mask = load_mask(mask_path(MASKS_DIR, job.image_id))
        p1.append(prepare(rgb, job.bbox_xyxy, "p1"))
        p3.append(prepare(rgb, job.bbox_xyxy, "p3", mask=mask))
    return [j.image_id for j in kept], p1, p3


def test_p3_embeddings_have_the_protocol_shape_and_are_unit_norm(encoder, p3_crops):
    _, _, crops = p3_crops
    out = encoder.embed(crops)
    assert out.shape == (P3_IMAGES, EXPECTED_DIM)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=NORM_TOL)


def test_p3_batch_matches_single_image_outputs(encoder, p3_crops):
    _, _, crops = p3_crops
    batched = encoder.embed(crops)
    singles = np.vstack([encoder.embed([c]) for c in crops])
    assert np.abs(batched - singles).max() < BATCH_TOL


def test_p3_identity_and_masking_change_the_embedding(encoder, p3_crops):
    _, p1, p3 = p3_crops
    identity = encoder.identity("p3")
    assert identity.variant == "p3"
    assert identity.dtype == "float16"
    assert identity.preprocessing_id == "dinov3_vit16_224_v1"
    cosine = (encoder.embed(p1) * encoder.embed(p3)).sum(axis=1)
    # the same image with and without the background: close but never identical
    assert (cosine < 0.9999).all()
    assert cosine.mean() > 0.5


def test_p3_save_then_load_is_exact(encoder, p3_crops, tmp_path):
    from avr.encoders.store import EmbeddingStore

    ids, _, crops = p3_crops
    out = encoder.embed(crops)
    store = EmbeddingStore(tmp_path, encoder.identity("p3"))
    store.write_shard(0, ids, out)

    loaded_ids, loaded = store.load_all()

    assert loaded_ids == ids
    np.testing.assert_array_equal(loaded, out)
    manifest = store.read_manifest(0)
    assert manifest.variant == "p3"
    assert manifest.revision == encoder.revision
    assert manifest.checkpoint_sha256 == encoder.checkpoint_sha256
