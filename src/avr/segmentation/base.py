"""Detector/segmenter interface (PROTOCOL.md §8, P2 and P3).

A ``Segmenter`` turns one RGB image into a list of ``Instance`` values, each with a box in
original-image pixels, a full-image boolean mask, a confidence and a class label. Every
adapter (RF-DETR today, any other detector later) converts its native output through
``instances_from_arrays`` so downstream code never sees a vendor type.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from PIL import Image

VEHICLE_LABELS = frozenset({"car", "truck", "bus"})


@dataclass(frozen=True)
class Instance:
    """One detected object: ``bbox_xyxy`` in image pixels, ``mask`` of shape (H, W), bool."""

    bbox_xyxy: tuple[float, float, float, float]
    mask: np.ndarray
    score: float
    label: str


@runtime_checkable
class Segmenter(Protocol):
    def segment(self, image: Image.Image) -> list[Instance]: ...

    def segment_batch(self, images: Sequence[Image.Image]) -> list[list[Instance]]: ...


def instances_from_arrays(
    xyxy: np.ndarray,
    scores: np.ndarray,
    labels: Sequence[str],
    masks: np.ndarray | None,
    keep_labels: Iterable[str] = VEHICLE_LABELS,
) -> list[Instance]:
    """Build instances from parallel arrays, keeping only ``keep_labels`` in detector order."""
    keep = set(keep_labels)
    n = len(labels)
    if n and masks is None:
        raise ValueError("segmentation output carries no mask array")
    instances = []
    for i in range(n):
        if labels[i] not in keep:
            continue
        x1, y1, x2, y2 = (float(v) for v in xyxy[i])
        instances.append(
            Instance(
                bbox_xyxy=(x1, y1, x2, y2),
                mask=np.asarray(masks[i]).astype(bool),
                score=float(scores[i]),
                label=str(labels[i]),
            )
        )
    return instances
