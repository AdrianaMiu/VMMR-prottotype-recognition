"""RF-DETR segmentation adapter (PROTOCOL.md §8 P2/P3; configs/segmentation/rfdetr_seg.yaml).

Wraps ``rfdetr.RFDETRSeg<Variant>`` behind ``avr.segmentation.base.Segmenter``. The model
is COCO-pretrained; only the vehicle labels listed in the config are kept. Weights are
downloaded by the package into ``$RF_HOME`` (default ``~/.roboflow/models``) and verified
here against the SHA-256 pinned in the config before any prediction.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from PIL import Image

from avr.io import sha256_of_file
from avr.segmentation.base import Instance, instances_from_arrays

VARIANT_CLASSES = {
    "nano": "RFDETRSegNano",
    "small": "RFDETRSegSmall",
    "medium": "RFDETRSegMedium",
    "large": "RFDETRSegLarge",
}


class CheckpointMismatch(RuntimeError):
    """The weight file on disk is not the one pinned in the registry."""


def verify_checkpoint(path: Path, expected_sha256: str) -> str:
    digest = sha256_of_file(Path(path))
    if digest != expected_sha256:
        raise CheckpointMismatch(f"{path}: sha256 {digest} != pinned {expected_sha256}")
    return digest


class RFDETRSegmenter:
    """Frozen RF-DETR instance segmenter; ``segment`` returns vehicle instances only."""

    def __init__(
        self,
        variant: str = "medium",
        threshold: float = 0.3,
        labels: Sequence[str] = ("car", "truck", "bus"),
        checkpoint_sha256: str | None = None,
        checkpoint_filename: str | None = None,
    ) -> None:
        import rfdetr
        from rfdetr.assets.model_weights import get_model_cache_dir

        self.threshold = float(threshold)
        self.labels = frozenset(labels)
        self.model = getattr(rfdetr, VARIANT_CLASSES[variant])()
        if checkpoint_sha256 is not None:
            filename = checkpoint_filename or self.model.model_config.pretrain_weights
            self.checkpoint_path = Path(get_model_cache_dir()) / Path(filename).name
            self.checkpoint_sha256 = verify_checkpoint(self.checkpoint_path, checkpoint_sha256)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> RFDETRSegmenter:
        return cls(
            variant=cfg.variant,
            threshold=cfg.threshold,
            labels=list(cfg.labels),
            checkpoint_sha256=cfg.checkpoint.sha256,
            checkpoint_filename=cfg.checkpoint.filename,
        )

    def segment(self, image: Image.Image) -> list[Instance]:
        return self.segment_batch([image])[0]

    def segment_batch(self, images: Sequence[Image.Image]) -> list[list[Instance]]:
        if not images:
            return []
        rgb = [image.convert("RGB") for image in images]
        detections = self.model.predict(rgb, threshold=self.threshold, include_source_image=False)
        if not isinstance(detections, list):
            detections = [detections]
        results = []
        for det in detections:
            names = det.data.get("class_name")
            labels = [str(n) for n in names] if names is not None else []
            results.append(
                instances_from_arrays(
                    np.asarray(det.xyxy),
                    np.asarray(det.confidence),
                    labels,
                    det.mask,
                    self.labels,
                )
            )
        return results
