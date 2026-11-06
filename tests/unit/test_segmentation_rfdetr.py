"""RF-DETR segmentation adapter registry (configs/segmentation/rfdetr_seg.yaml)."""

import re
from pathlib import Path

import pytest

from avr.config import load_config
from avr.segmentation.rfdetr_seg import CheckpointMismatch, verify_checkpoint

SHA256_ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_segmentation_group_is_part_of_the_default_composition():
    cfg = load_config()

    assert cfg.segmentation.name == "rfdetr_seg"
    assert cfg.segmentation.variant == "medium"
    assert cfg.segmentation.checkpoint.filename == "rf-detr-seg-medium.pt"
    assert cfg.segmentation.checkpoint.url.startswith("https://")
    assert re.fullmatch(r"[0-9a-f]{64}", cfg.segmentation.checkpoint.sha256)
    assert re.fullmatch(r"[0-9a-f]{32}", cfg.segmentation.checkpoint.md5)
    assert cfg.segmentation.licence == "Apache-2.0"
    assert set(cfg.segmentation.labels) == {"car", "truck", "bus"}
    assert 0 < cfg.segmentation.threshold < 1


def test_verify_checkpoint_accepts_a_matching_digest(tmp_path: Path):
    path = tmp_path / "w.pt"
    path.write_bytes(b"abc")

    assert verify_checkpoint(path, SHA256_ABC) == SHA256_ABC


def test_verify_checkpoint_rejects_a_different_file(tmp_path: Path):
    path = tmp_path / "w.pt"
    path.write_bytes(b"abd")

    with pytest.raises(CheckpointMismatch):
        verify_checkpoint(path, SHA256_ABC)
