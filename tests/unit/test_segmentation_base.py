"""Segmenter interface (PROTOCOL.md §8 P2/P3): instances with a full-image mask, score, label."""

import numpy as np
import pytest

from avr.segmentation.base import Instance, instances_from_arrays


def _masks(n: int, shape=(4, 6)) -> np.ndarray:
    masks = np.zeros((n, *shape), dtype=bool)
    for i in range(n):
        masks[i, i % shape[0], :] = True
    return masks


def test_instances_keep_only_vehicle_labels_in_detector_order():
    xyxy = np.array([[0, 0, 2, 2], [1, 1, 3, 3], [2, 2, 4, 4]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    labels = ["car", "person", "truck"]

    instances = instances_from_arrays(xyxy, scores, labels, _masks(3), keep_labels={"car", "truck"})

    assert [i.label for i in instances] == ["car", "truck"]
    assert instances[0].bbox_xyxy == (0.0, 0.0, 2.0, 2.0)
    assert instances[1].score == pytest.approx(0.7)
    assert instances[1].mask.dtype == bool and instances[1].mask.shape == (4, 6)
    assert instances[1].mask[2].all() and not instances[1].mask[0].any()


def test_no_detections_gives_an_empty_list():
    assert instances_from_arrays(np.zeros((0, 4)), np.zeros(0), [], None, {"car"}) == []


def test_missing_masks_raise_rather_than_silently_producing_empty_masks():
    with pytest.raises(ValueError, match="mask"):
        instances_from_arrays(np.zeros((1, 4)), np.ones(1), ["car"], None, {"car"})


def test_instance_is_a_plain_value_object():
    inst = Instance(
        bbox_xyxy=(0.0, 0.0, 1.0, 1.0), mask=np.ones((2, 2), bool), score=0.5, label="car"
    )

    assert inst.mask.sum() == 4
