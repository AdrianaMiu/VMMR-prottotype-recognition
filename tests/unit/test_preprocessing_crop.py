"""P1 crop geometry (PROTOCOL.md §8): 5 % expansion on each side, clipped, aspect preserved.

Expected boxes are worked out by hand from the protocol rule.
"""

from PIL import Image

from avr.preprocessing.crop import crop_with_padding, p1_crop_box


def test_box_expands_five_percent_of_its_own_size_on_each_side():
    # width 200 -> 10 px each side, height 100 -> 5 px each side
    assert p1_crop_box((100, 100, 300, 200), width=800, height=600) == (90, 95, 310, 205)


def test_fractional_expansion_rounds_outwards():
    # width 11 -> 0.55 px: floor on the near side, ceil on the far side
    assert p1_crop_box((10, 10, 21, 21), width=100, height=100) == (9, 9, 22, 22)


def test_expansion_is_clipped_to_the_image_bounds():
    assert p1_crop_box((0, 0, 800, 600), width=800, height=600) == (0, 0, 800, 600)
    assert p1_crop_box((790, 590, 800, 600), width=800, height=600) == (789, 589, 800, 600)


def test_custom_expansion_fraction():
    assert p1_crop_box((100, 100, 300, 200), width=800, height=600, expand=0.5) == (
        0,
        50,
        400,
        250,
    )


def test_crop_with_padding_keeps_the_box_size_without_resizing():
    image = Image.new("RGB", (800, 600), (10, 20, 30))

    crop = crop_with_padding(image, (90, 95, 310, 205))

    assert crop.size == (220, 110)
    assert crop.getpixel((0, 0)) == (10, 20, 30)


def test_crop_with_padding_fills_pixels_outside_the_image_with_neutral_grey():
    image = Image.new("RGB", (20, 20), (255, 0, 0))

    crop = crop_with_padding(image, (-5, 0, 20, 20))

    assert crop.size == (25, 20)
    assert crop.getpixel((0, 0)) == (127, 127, 127)
    assert crop.getpixel((5, 0)) == (255, 0, 0)
