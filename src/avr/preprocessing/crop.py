"""P1 crop geometry (PROTOCOL.md §8).

P1 is the CompCars ground-truth bounding box expanded by 5 % of its own width and height
on each side and clipped to the image. The crop is never resized here: letterboxing to the
encoder input (aspect preserved with encoder-specific padding) is the encoder's job. P3
reuses exactly this geometry so the two variants differ only by masking.
"""

from __future__ import annotations

import math

from PIL import Image

NEUTRAL_RGB = (127, 127, 127)
BoxXYXY = tuple[int, int, int, int]


def p1_crop_box(
    bbox_xyxy: tuple[int, int, int, int], width: int, height: int, expand: float = 0.05
) -> BoxXYXY:
    """Expand ``bbox_xyxy`` by ``expand`` of its size on every side and clip to the image.

    Fractional margins round outwards (floor on x1/y1, ceil on x2/y2) so the box never
    loses ground-truth pixels.
    """
    x1, y1, x2, y2 = bbox_xyxy
    dx = (x2 - x1) * expand
    dy = (y2 - y1) * expand
    return (
        max(0, math.floor(x1 - dx)),
        max(0, math.floor(y1 - dy)),
        min(width, math.ceil(x2 + dx)),
        min(height, math.ceil(y2 + dy)),
    )


def crop_with_padding(
    image: Image.Image, box: BoxXYXY, fill: tuple[int, int, int] = NEUTRAL_RGB
) -> Image.Image:
    """Crop ``box`` from ``image`` at native resolution; pixels outside the image get ``fill``.

    The output size is exactly the box size, so aspect ratio is preserved by construction.
    """
    x1, y1, x2, y2 = box
    canvas = Image.new("RGB", (x2 - x1, y2 - y1), fill)
    ix1, iy1 = max(x1, 0), max(y1, 0)
    ix2, iy2 = min(x2, image.width), min(y2, image.height)
    if ix2 > ix1 and iy2 > iy1:
        canvas.paste(image.convert("RGB").crop((ix1, iy1, ix2, iy2)), (ix1 - x1, iy1 - y1))
    return canvas
