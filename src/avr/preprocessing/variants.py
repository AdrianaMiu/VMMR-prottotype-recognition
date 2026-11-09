"""Preprocessing variants as one entry point (PROTOCOL.md §8, GLOSSARY.md).

``prepare`` returns the PIL crop an encoder embeds for a given variant. P1 is the expanded
ground-truth box crop (``avr.preprocessing.crop``); P3 is the same geometry with the
background replaced by ``NEUTRAL_RGB`` using a stored full-image mask (``avr.preprocessing
.p3``). The two variants therefore differ only by masking, never by geometry, which is what
makes the P1 vs P3 comparison an ablation of background rather than of crop size.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from PIL import Image

from avr.preprocessing.crop import NEUTRAL_RGB, crop_with_padding, p1_crop_box
from avr.preprocessing.p3 import project_mask

Variant = Literal["p1", "p3"]
VARIANTS: tuple[Variant, ...] = ("p1", "p3")


def prepare(
    image: Image.Image,
    bbox_xyxy: tuple[int, int, int, int],
    variant: Variant,
    mask: np.ndarray | None = None,
    expand: float = 0.05,
    background: tuple[int, int, int] = NEUTRAL_RGB,
) -> Image.Image:
    """Crop ``image`` for ``variant``; P3 needs the full-image boolean ``mask``."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    box = p1_crop_box(bbox_xyxy, image.width, image.height, expand=expand)
    crop = crop_with_padding(image, box, fill=background)
    if variant == "p1":
        return crop
    if mask is None:
        raise ValueError("variant 'p3' requires a full-image mask")
    if mask.shape != (image.height, image.width):
        raise ValueError(f"mask shape {mask.shape} does not match image {image.size[::-1]}")
    keep = project_mask(np.asarray(mask, dtype=bool), box)
    arr = np.asarray(crop).copy()
    arr[~keep] = background
    return Image.fromarray(arr, "RGB")
