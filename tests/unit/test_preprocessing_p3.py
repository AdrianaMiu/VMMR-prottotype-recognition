"""P3 segmented vehicle on a neutral background, projected into the P1 crop (PROTOCOL.md §8).

Synthetic 100x100 image, GT box (10, 10, 50, 50). Expected numbers are hand-calculated:
P1 crop = (8, 8, 52, 52) -> 44x44 = 1936 px; instance box (12, 12, 48, 48) has
IoU 1296/1600 = 0.81 with the GT box; a 20x20 mask covers 400/1936 of the crop.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from avr.preprocessing.crop import crop_with_padding, p1_crop_box
from avr.preprocessing.p3 import (
    NEUTRAL_RGB,
    box_iou,
    build_p3,
    load_mask,
    mask_path,
    save_mask,
    select_instance,
)
from avr.segmentation.base import Instance

GT = (10, 10, 50, 50)


def _image() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, size=(100, 100, 3), dtype=np.uint8), "RGB")


def _instance(box, score, rows=slice(20, 40), cols=slice(20, 40), label="car") -> Instance:
    mask = np.zeros((100, 100), dtype=bool)
    mask[rows, cols] = True
    return Instance(bbox_xyxy=tuple(float(v) for v in box), mask=mask, score=score, label=label)


def test_box_iou_hand_values():
    assert box_iou(GT, (12, 12, 48, 48)) == pytest.approx(0.81)
    assert box_iou(GT, (50, 50, 90, 90)) == 0.0
    assert box_iou(GT, GT) == 1.0


def test_select_instance_prefers_gt_overlap_over_detector_score():
    good = _instance((12, 12, 48, 48), score=0.4)
    distractor = _instance((40, 40, 90, 90), score=0.99)

    chosen, iou = select_instance([distractor, good], GT)

    assert chosen is good
    assert iou == pytest.approx(0.81)


def test_select_instance_returns_none_without_detections():
    assert select_instance([], GT) is None


def test_p3_has_exactly_the_p1_geometry_and_differs_only_by_masking():
    image = _image()
    result = build_p3(image, GT, [_instance((12, 12, 48, 48), score=0.9)])
    p1 = crop_with_padding(image, p1_crop_box(GT, 100, 100))

    assert result.crop_box == (8, 8, 52, 52)
    assert result.p1.size == result.p3.size == p1.size == (44, 44)
    assert result.mask.shape == (44, 44)
    p1_arr, p3_arr = np.asarray(p1), np.asarray(result.p3)
    assert np.array_equal(np.asarray(result.p1), p1_arr)
    assert np.array_equal(p3_arr[result.mask], p1_arr[result.mask])
    assert (p3_arr[~result.mask] == NEUTRAL_RGB).all()
    assert result.p3.mode == "RGB"  # never transparency


def test_ok_result_records_iou_area_fraction_and_score():
    result = build_p3(_image(), GT, [_instance((12, 12, 48, 48), score=0.9)])

    assert result.status == "ok"
    assert result.gt_iou == pytest.approx(0.81)
    assert result.mask_area_fraction == pytest.approx(400 / 1936)
    assert result.detection_score == pytest.approx(0.9)
    # the 20x20 mask at image rows/cols 20..40 lands at crop rows/cols 12..32
    assert result.mask[12:32, 12:32].all() and result.mask.sum() == 400


def test_no_detection_falls_back_to_the_unmasked_p1_crop():
    result = build_p3(_image(), GT, [])

    assert result.status == "no_detection"
    assert result.gt_iou is None and result.detection_score is None
    assert result.mask_area_fraction is None
    assert np.array_equal(np.asarray(result.p3), np.asarray(result.p1))
    assert result.mask.all()


def test_low_overlap_keeps_the_iou_but_does_not_mask():
    result = build_p3(_image(), GT, [_instance((40, 40, 90, 90), score=0.99)], min_iou=0.5)

    assert result.status == "low_overlap"
    assert 0 < result.gt_iou < 0.5
    assert result.detection_score == pytest.approx(0.99)
    assert np.array_equal(np.asarray(result.p3), np.asarray(result.p1))


def test_mask_round_trips_losslessly_through_png(tmp_path: Path):
    mask = np.zeros((44, 44), dtype=bool)
    mask[3:7, 10:30] = True
    path = mask_path(tmp_path, "0123456789abcdef")

    save_mask(mask, path)

    assert path == tmp_path / "0123456789abcdef.png"
    assert path.suffix == ".png" and path.exists()
    assert np.array_equal(load_mask(path), mask)
    assert load_mask(path).dtype == bool
