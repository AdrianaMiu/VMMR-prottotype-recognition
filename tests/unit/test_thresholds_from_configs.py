"""The P1 expansion and the P3 low-overlap threshold live in the configs, not in code.

Expected values are read from the YAML files here with ``yaml.safe_load`` (independently
of the modules under test) and the Python source is scanned for stray literals.
"""

from pathlib import Path

import yaml

from avr.config import CONFIG_DIR, load_config
from avr.preprocessing.crop import P1_EXPAND_FRACTION
from avr.preprocessing.p3 import DEFAULT_MIN_IOU

SRC = Path(__file__).resolve().parents[2] / "src" / "avr"


def test_p1_expand_fraction_is_the_config_value():
    expected = yaml.safe_load((CONFIG_DIR / "preprocessing" / "p1_bbox_crop.yaml").read_text())
    assert P1_EXPAND_FRACTION == expected["bbox_expand_fraction"]
    assert load_config().preprocessing.bbox_expand_fraction == P1_EXPAND_FRACTION


def test_min_iou_is_the_segmentation_config_value():
    expected = yaml.safe_load((CONFIG_DIR / "segmentation" / "rfdetr_seg.yaml").read_text())
    assert DEFAULT_MIN_IOU == expected["min_iou"]
    assert load_config().segmentation.min_iou == DEFAULT_MIN_IOU


def test_no_literal_expand_default_remains_in_the_package():
    offenders = [
        p.relative_to(SRC).as_posix()
        for p in SRC.rglob("*.py")
        if "expand: float = 0.05" in p.read_text()
    ]
    assert offenders == []
