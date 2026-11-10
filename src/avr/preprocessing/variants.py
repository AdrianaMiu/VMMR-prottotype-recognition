"""Preprocessing variants as one entry point (PROTOCOL.md §8, GLOSSARY.md).

``prepare`` returns the PIL crop an encoder embeds for a given variant. P1 is the expanded
ground-truth box crop (``avr.preprocessing.crop``); P3 is the same geometry with the
background replaced by ``NEUTRAL_RGB``. The mask may be given either in full-image
coordinates (as a segmenter emits it) or already in P1-crop geometry, which is how
``scripts/build_p3_masks.py`` stores it under ``data/masks/`` (``avr.preprocessing.p3``).
The two variants therefore differ only by masking, never by geometry, which is what makes
the P1 vs P3 comparison an ablation of background rather than of crop size.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from PIL import Image

from avr.preprocessing.crop import NEUTRAL_RGB, P1_EXPAND_FRACTION, crop_with_padding, p1_crop_box
from avr.preprocessing.p3 import project_mask

Variant = Literal["p1", "p3"]
VARIANTS: tuple[Variant, ...] = ("p1", "p3")


def prepare(
    image: Image.Image,
    bbox_xyxy: tuple[int, int, int, int],
    variant: Variant,
    mask: np.ndarray | None = None,
    expand: float = P1_EXPAND_FRACTION,
    background: tuple[int, int, int] = NEUTRAL_RGB,
) -> Image.Image:
    """Crop ``image`` for ``variant``; P3 needs a boolean ``mask`` (full-image or crop-sized)."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    box = p1_crop_box(bbox_xyxy, image.width, image.height, expand=expand)
    crop = crop_with_padding(image, box, fill=background)
    if variant == "p1":
        return crop
    if mask is None:
        raise ValueError("variant 'p3' requires a full-image mask")
    mask = np.asarray(mask, dtype=bool)
    crop_shape = (crop.height, crop.width)
    if mask.shape == crop_shape:
        keep = mask
    elif mask.shape == (image.height, image.width):
        keep = project_mask(mask, box)
    else:
        raise ValueError(
            f"mask shape {mask.shape} matches neither the image {(image.height, image.width)} "
            f"nor the P1 crop {crop_shape}"
        )
    arr = np.asarray(crop).copy()
    arr[~keep] = background
    return Image.fromarray(arr, "RGB")
