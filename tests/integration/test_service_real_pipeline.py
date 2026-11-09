"""The real pipeline (RF-DETR + DINOv3 ViT-B/16 + M1 gate prototypes) on three CompCars images.

Skipped without a CUDA device, the pinned RF-DETR checkpoint, the local dataset or the
M1 gate prototype tables and embedded Qdrant store under ``data/``. The Track A class of a
CompCars image is ``<make_id:03d>_<model_id:04d>`` read from its path, so the expected top-1
comes from the dataset layout, not from the code under test.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from avr.config import load_config

pytest.importorskip("torch")
pytest.importorskip("rfdetr")

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE = "track_a__dinov3_vitb16__p1__native__m0"
# web-nature images of three Track A classes (make_id/model_id/year/<id>.jpg)
IMAGES = [
    "image/78/27/2009/6e8c142af63a1e.jpg",
    "image/119/1819/2014/8c99f37820c58b.jpg",
    "image/119/1819/2014/408f6d07ccdfe6.jpg",
]


def track_a_class_id(relative_path: str) -> str:
    _, make_id, model_id, *_ = relative_path.split("/")
    return f"{int(make_id):03d}_{int(model_id):04d}"


@pytest.fixture(scope="module")
def resources():
    import torch

    from avr.service.app import ServiceResources
    from avr.service.settings import ServiceSettings

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    cfg = load_config().segmentation
    cache = Path(os.environ.get("RF_HOME", "~/.roboflow/models")).expanduser()
    if not (cache / cfg.checkpoint.filename).exists():
        pytest.skip("RF-DETR checkpoint not downloaded")
    settings = ServiceSettings()
    if not (settings.prototypes_dir / "dinov3_vitb16" / "p1").is_dir():
        pytest.skip("M1 gate prototype tables not available under data/prototypes")
    if not settings.qdrant_dir.is_dir():
        pytest.skip("embedded Qdrant store not available under data/qdrant")
    r = ServiceResources(settings)
    r.load()
    return r


@pytest.fixture(scope="module")
def dataset_root() -> Path:
    relative = load_config().dataset.root
    # the config path is relative to the main checkout; a worktree sits two levels deeper
    for base in (REPO_ROOT, REPO_ROOT.parents[1]):
        root = (base / relative).resolve()
        if (root / "image").is_dir():
            return root
    pytest.skip("CompCars release not available")


def test_real_pipeline_recognises_at_least_one_of_three_images(resources, dataset_root):
    from PIL import Image

    profile = resources.profiles[PROFILE]
    pipeline = resources.pipeline_for(profile)
    top1_correct = 0
    for rel in IMAGES:
        with Image.open(dataset_root / rel) as im:
            result = pipeline.recognize(im.convert("RGB"), top_k=5)
        assert result.vehicles, f"{rel}: no vehicle detected"
        for vehicle in result.vehicles:
            assert len(vehicle.predictions) == 5
            assert all(v >= 0 for v in vehicle.latency_ms.model_dump().values())
        best = max(result.vehicles, key=lambda v: v.detection_score)
        top1_correct += best.predictions[0].class_id == track_a_class_id(rel)
    assert top1_correct >= 1
