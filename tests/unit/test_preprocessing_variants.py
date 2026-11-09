"""``variants.prepare``: one entry point that yields the P1 or P3 crop for an image."""

import numpy as np
import pytest
from PIL import Image

from avr.preprocessing.variants import VARIANTS, prepare

# 40x30 image, GT box (10, 5, 30, 25): 20x20, 5 % margin = 1 px, so the P1 box is
# (9, 4, 31, 26) and the crop is 22x22 (hand-computed from PROTOCOL.md §8).
BBOX = (10, 5, 30, 25)


def _image() -> Image.Image:
    return Image.new("RGB", (40, 30), (200, 10, 10))


def _mask() -> np.ndarray:
    mask = np.zeros((30, 40), dtype=bool)
    mask[10:20, 15:25] = True  # vehicle pixels only in the centre 10x10 block
    return mask


def test_variants_are_exactly_p1_and_p3():
    assert VARIANTS == ("p1", "p3")


def test_p1_crop_has_the_expanded_box_size_and_original_pixels():
    crop = prepare(_image(), BBOX, "p1")
    assert crop.size == (22, 22)
    assert crop.getpixel((0, 0)) == (200, 10, 10)


def test_p3_crop_has_the_same_size_and_neutral_background():
    p1 = prepare(_image(), BBOX, "p1")
    p3 = prepare(_image(), BBOX, "p3", mask=_mask())
    assert p3.size == p1.size
    arr = np.asarray(p3)
    # mask block (15..25, 10..20) in image pixels -> (6..16, 6..16) in crop pixels
    assert (arr[6:16, 6:16] == (200, 10, 10)).all()
    assert tuple(arr[0, 0]) == (127, 127, 127)
    assert tuple(arr[21, 21]) == (127, 127, 127)
    background = np.ones((22, 22), dtype=bool)
    background[6:16, 6:16] = False
    assert (arr[background] == (127, 127, 127)).all()


def test_p3_accepts_the_stored_crop_geometry_mask():
    """``scripts/build_p3_masks.py`` stores masks already projected into the P1 crop."""
    crop_mask = np.zeros((22, 22), dtype=bool)
    crop_mask[6:16, 6:16] = True
    from_full = prepare(_image(), BBOX, "p3", mask=_mask())
    from_crop = prepare(_image(), BBOX, "p3", mask=crop_mask)
    np.testing.assert_array_equal(np.asarray(from_crop), np.asarray(from_full))


def test_p3_rejects_a_mask_of_unrelated_shape():
    with pytest.raises(ValueError, match="mask shape"):
        prepare(_image(), BBOX, "p3", mask=np.ones((5, 5), dtype=bool))


def test_p3_requires_a_mask():
    with pytest.raises(ValueError, match="mask"):
        prepare(_image(), BBOX, "p3")


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="variant"):
        prepare(_image(), BBOX, "p2")  # type: ignore[arg-type]
