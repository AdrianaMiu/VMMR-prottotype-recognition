"""RF-DETR masks on five real CompCars images (issue M1-19 acceptance).

Skipped without a CUDA device, the pinned checkpoint, or the local dataset.
"""

import os
from pathlib import Path

import pytest

from avr.config import load_config
from avr.preprocessing.p3 import build_p3, load_mask, save_mask

pytest.importorskip("torch")
pytest.importorskip("rfdetr")

REPO_ROOT = Path(__file__).resolve().parents[2]
# (relative image path, GT bbox from the label file) for five web-nature images
SAMPLES = [
    ("image/78/27/2009/6e8c142af63a1e.jpg", (181, 211, 724, 572)),
    ("image/119/1819/2014/8c99f37820c58b.jpg", (85, 124, 707, 421)),
    ("image/119/1819/2014/408f6d07ccdfe6.jpg", (62, 174, 741, 403)),
    ("image/119/1819/2014/d10351f2f24011.jpg", (106, 137, 793, 541)),
    ("image/1/1101/2011/07b90decb92ba6.jpg", None),
]


@pytest.fixture(scope="module")
def segmenter():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    cfg = load_config().segmentation
    cache = Path(os.environ.get("RF_HOME", "~/.roboflow/models")).expanduser()
    if not (cache / cfg.checkpoint.filename).exists():
        pytest.skip("RF-DETR checkpoint not downloaded")
    from avr.segmentation.rfdetr_seg import RFDETRSegmenter

    return RFDETRSegmenter.from_config(cfg)


@pytest.fixture(scope="module")
def dataset_root() -> Path:
    root = (REPO_ROOT / load_config().dataset.root).resolve()
    if not (root / "image").is_dir():
        pytest.skip("CompCars release not available")
    return root


def test_five_real_images_get_ok_masks(segmenter, dataset_root: Path, tmp_path: Path):
    from PIL import Image

    from avr.data.labels import parse_label

    statuses = []
    for rel, bbox in SAMPLES:
        if bbox is None:
            bbox = parse_label(
                dataset_root / rel.replace("image/", "label/", 1).replace(".jpg", ".txt")
            ).bbox_xyxy
        image = Image.open(dataset_root / rel).convert("RGB")
        result = build_p3(image, bbox, segmenter.segment(image))
        statuses.append(result.status)
        assert result.p1.size == result.p3.size
        assert 0.2 < result.mask_area_fraction < 1.0
        path = tmp_path / f"{Path(rel).stem}.png"
        save_mask(result.mask, path)
        assert (load_mask(path) == result.mask).all()
    assert statuses == ["ok"] * 5
