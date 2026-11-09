"""Recognition pipeline on a raw scene image (PROTOCOL.md §16 "Pipeline").

``input image -> vehicle detection and instance segmentation (one RF-DETR pass) -> P1/P3
crop per vehicle -> one batched encoder call -> prototype search -> ranked results per
vehicle``. The three collaborators are injected through narrow seams so the pipeline runs
without a GPU in tests: a ``Segmenter`` (``avr.segmentation.base``), an ``Embedder`` with the
``FrozenEncoder.embed`` contract and a ``SearchBackend`` (``avr.service.search``).

Coordinates: boxes are reported exactly as the detector emitted them, in input-image pixels;
the crop geometry is the P1 rule of ``avr.preprocessing.crop`` applied to the box rounded
outwards to integers, so P1 and P3 share one geometry and differ only by masking.

Timings: ``detection`` is the segmenter call (detection and masks come out of one forward
pass), ``segmentation`` the mask projection and crop preparation, ``embedding`` the one
batched encoder call and ``search`` the prototype search for the batch. The last two are
batch-level and are reported unchanged on every vehicle of the image.
"""

from __future__ import annotations

import base64
import io
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from PIL import Image

from avr.preprocessing.variants import Variant, prepare
from avr.segmentation.base import Instance, Segmenter
from avr.service.schemas import Decision, Prediction, StageLatency, Vehicle
from avr.service.search import Hit, SearchBackend


class Embedder(Protocol):
    """The part of :class:`avr.encoders.adapter.FrozenEncoder` the pipeline uses."""

    @property
    def native_dim(self) -> int: ...

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray: ...


@dataclass
class PipelineResult:
    vehicles: list[Vehicle]
    latency_ms: dict[str, float] = field(default_factory=dict)
    image_size: tuple[int, int] = (0, 0)


def integer_box(
    bbox_xyxy: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    """Round a detector box outwards to whole pixels and clip it to the image."""
    x1, y1, x2, y2 = bbox_xyxy
    return (
        max(0, math.floor(x1)),
        max(0, math.floor(y1)),
        min(width, math.ceil(x2)),
        min(height, math.ceil(y2)),
    )


def mask_to_png_base64(mask: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, "L").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _ms(start: float) -> float:
    return max(0.0, (time.perf_counter() - start) * 1000.0)


class RecognitionPipeline:
    def __init__(
        self,
        segmenter: Segmenter,
        encoder: Embedder,
        search: SearchBackend,
        variant: Variant,
        accept_threshold: float = 0.6,
        min_detection_score: float = 0.0,
        expand: float = 0.05,
    ) -> None:
        self.segmenter = segmenter
        self.encoder = encoder
        self.search = search
        self.variant: Variant = variant
        self.accept_threshold = float(accept_threshold)
        self.min_detection_score = float(min_detection_score)
        self.expand = float(expand)

    def detect(self, image: Image.Image) -> list[Instance]:
        instances = self.segmenter.segment(image)
        return [inst for inst in instances if inst.score >= self.min_detection_score]

    def crop(self, image: Image.Image, instance: Instance) -> Image.Image:
        box = integer_box(instance.bbox_xyxy, image.width, image.height)
        return prepare(image, box, self.variant, mask=instance.mask, expand=self.expand)

    def recognize(
        self,
        image: Image.Image,
        top_k: int = 10,
        return_masks: bool = False,
        return_embeddings: bool = False,
    ) -> PipelineResult:
        image = image.convert("RGB")
        t0 = time.perf_counter()
        instances = self.detect(image)
        detection_ms = _ms(t0)

        t1 = time.perf_counter()
        crops = [self.crop(image, inst) for inst in instances]
        segmentation_ms = _ms(t1)

        t2 = time.perf_counter()
        if crops:
            embeddings = self.encoder.embed(crops)
        else:
            embeddings = np.zeros((0, self.encoder.native_dim), dtype=np.float32)
        embedding_ms = _ms(t2)

        t3 = time.perf_counter()
        hits = self.search.search(embeddings, int(top_k)) if crops else []
        search_ms = _ms(t3)

        latency = StageLatency(
            detection=detection_ms,
            segmentation=segmentation_ms,
            embedding=embedding_ms,
            search=search_ms,
        )
        vehicles = [
            self._vehicle(
                i,
                inst,
                hits[i],
                latency,
                mask_to_png_base64(inst.mask) if return_masks else None,
                embeddings[i].tolist() if return_embeddings else None,
            )
            for i, inst in enumerate(instances)
        ]
        return PipelineResult(
            vehicles=vehicles,
            latency_ms=latency.model_dump(),
            image_size=(image.width, image.height),
        )

    def _vehicle(
        self,
        detection_id: int,
        instance: Instance,
        hits: Sequence[Hit],
        latency: StageLatency,
        mask_png: str | None,
        embedding: list[float] | None,
    ) -> Vehicle:
        predictions = [
            Prediction(
                rank=rank,
                make=hit.payload.get("make_name"),
                model=hit.payload.get("model_name"),
                generation=hit.payload.get("generation_id"),
                class_id=hit.class_id,
                cosine_similarity=float(np.clip(hit.score, -1.0, 1.0)),
                calibrated_confidence=None,
            )
            for rank, hit in enumerate(hits, start=1)
        ]
        return Vehicle(
            detection_id=detection_id,
            bbox_xyxy=[float(v) for v in instance.bbox_xyxy],
            detection_score=float(np.clip(instance.score, 0.0, 1.0)),
            viewpoint=None,
            predictions=predictions,
            decision=self._decision(predictions),
            latency_ms=latency,
            mask_png_base64=mask_png,
            embedding=embedding,
        )

    def _decision(self, predictions: Sequence[Prediction]) -> Decision:
        if not predictions:
            return "no_match"
        if predictions[0].cosine_similarity >= self.accept_threshold:
            return "accepted"
        return "uncertain"
