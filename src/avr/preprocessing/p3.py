"""P3: segmented vehicle on a neutral background inside the P1 crop (PROTOCOL.md §8).

For one image: pick the predicted instance whose box best overlaps the CompCars ground-truth
box (IoU), project its full-image mask into the exact P1 crop geometry (``avr.preprocessing
.crop``), and replace the background with ``NEUTRAL_RGB``. Never transparency. The binary
mask is stored separately as a PNG named by ``image_id``.

``mask_status`` makes the failure rate reportable: ``ok`` (masked), ``no_detection`` (no
vehicle instance at all) and ``low_overlap`` (best IoU below ``min_iou``). Failed images
keep the unmasked P1 crop and an all-true mask so every image stays traceable; downstream
experiments filter on the status.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from PIL import Image

from avr.config import CONFIG_DIR
from avr.preprocessing.crop import (
    NEUTRAL_RGB,
    P1_EXPAND_FRACTION,
    BoxXYXY,
    crop_with_padding,
    p1_crop_box,
)
from avr.segmentation.base import Instance

MaskStatus = Literal["ok", "no_detection", "low_overlap"]
MASK_STATUSES: tuple[MaskStatus, ...] = ("ok", "no_detection", "low_overlap")
SEGMENTATION_CONFIG = CONFIG_DIR / "segmentation" / "rfdetr_seg.yaml"
# The one source of the low-overlap threshold; scripts read the same key through Hydra.
DEFAULT_MIN_IOU: float = float(yaml.safe_load(SEGMENTATION_CONFIG.read_text())["min_iou"])


@dataclass(frozen=True)
class P3Result:
    crop_box: BoxXYXY
    p1: Image.Image
    p3: Image.Image
    mask: np.ndarray
    status: MaskStatus
    gt_iou: float | None
    mask_area_fraction: float | None
    detection_score: float | None


def box_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def select_instance(
    instances: list[Instance], gt_bbox: tuple[int, int, int, int]
) -> tuple[Instance, float] | None:
    """Instance with the highest box IoU against the ground-truth box; ties go to the score."""
    if not instances:
        return None
    best = max(instances, key=lambda inst: (box_iou(gt_bbox, inst.bbox_xyxy), inst.score))
    return best, box_iou(gt_bbox, best.bbox_xyxy)


def project_mask(mask: np.ndarray, box: BoxXYXY) -> np.ndarray:
    """Cut the full-image mask to ``box``; pixels outside the image count as background."""
    x1, y1, x2, y2 = box
    out = np.zeros((y2 - y1, x2 - x1), dtype=bool)
    h, w = mask.shape
    ix1, iy1, ix2, iy2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    if ix2 > ix1 and iy2 > iy1:
        out[iy1 - y1 : iy2 - y1, ix1 - x1 : ix2 - x1] = mask[iy1:iy2, ix1:ix2]
    return out


def build_p3(
    image: Image.Image,
    gt_bbox: tuple[int, int, int, int],
    instances: list[Instance],
    expand: float = P1_EXPAND_FRACTION,
    min_iou: float = DEFAULT_MIN_IOU,
    background: tuple[int, int, int] = NEUTRAL_RGB,
) -> P3Result:
    box = p1_crop_box(gt_bbox, image.width, image.height, expand=expand)
    p1 = crop_with_padding(image, box)
    selected = select_instance(instances, gt_bbox)
    if selected is None:
        return P3Result(
            box, p1, p1.copy(), np.ones(p1.size[::-1], bool), "no_detection", None, None, None
        )
    instance, iou = selected
    if iou < min_iou:
        return P3Result(
            box,
            p1,
            p1.copy(),
            np.ones(p1.size[::-1], bool),
            "low_overlap",
            iou,
            None,
            instance.score,
        )
    mask = project_mask(instance.mask, box)
    p3_arr = np.asarray(p1).copy()
    p3_arr[~mask] = background
    p3 = Image.fromarray(p3_arr, "RGB")
    return P3Result(box, p1, p3, mask, "ok", iou, float(mask.mean()), instance.score)


def mask_path(masks_dir: Path, image_id: str) -> Path:
    return Path(masks_dir) / f"{image_id}.png"


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, "L").save(path, format="PNG", optimize=True)


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L")) > 127
