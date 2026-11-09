"""Fakes for the pipeline seams: a scripted segmenter and a colour-reading encoder (no GPU)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image

from avr.segmentation.base import Instance


class ScriptedSegmenter:
    """Returns the instances given at construction, in order, for every image."""

    def __init__(self, instances: list[Instance]) -> None:
        self.instances = instances
        self.calls = 0

    def segment(self, image: Image.Image) -> list[Instance]:
        self.calls += 1
        return list(self.instances)

    def segment_batch(self, images: Sequence[Image.Image]) -> list[list[Instance]]:
        return [self.segment(im) for im in images]


class ColourEncoder:
    """Embeds a crop as the unit vector of its dominant mean channel (R->axis 0, G->1, B->2).

    Pure grey crops (neutral background only) embed on axis 4, so a crop whose vehicle pixels
    were all masked away is distinguishable from any colour.
    """

    name = "toy"
    native_dim = 5
    revision = "r1"
    checkpoint_sha256 = "0" * 64
    preprocessing_id = "toy_pre"

    def __init__(self) -> None:
        self.calls: list[list[Image.Image]] = []

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        self.calls.append(list(images))
        out = np.zeros((len(images), self.native_dim), dtype=np.float32)
        for i, im in enumerate(images):
            mean = np.asarray(im.convert("RGB"), dtype=np.float64).reshape(-1, 3).mean(axis=0)
            if np.ptp(mean) < 1.0:
                out[i, 4] = 1.0
            else:
                out[i, int(np.argmax(mean))] = 1.0
        return out


def box_instance(
    bbox: tuple[float, float, float, float], size: tuple[int, int], score: float = 0.9
) -> Instance:
    """A ``car`` instance whose mask is exactly the box (``size`` is the image (W, H))."""
    w, h = size
    mask = np.zeros((h, w), dtype=bool)
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    mask[y1:y2, x1:x2] = True
    return Instance(bbox_xyxy=bbox, mask=mask, score=score, label="car")


def scene(size: tuple[int, int], boxes: dict[tuple[int, int, int, int], tuple[int, int, int]]):
    """A grey canvas with one solid colour rectangle per box."""
    image = Image.new("RGB", size, (127, 127, 127))
    for (x1, y1, x2, y2), colour in boxes.items():
        image.paste(Image.new("RGB", (x2 - x1, y2 - y1), colour), (x1, y1))
    return image
