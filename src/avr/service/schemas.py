"""Request and response schemas of the recognition service (PROTOCOL.md §16).

The response follows the §16 output skeleton field for field and in field order, so the
JSON a client sees is deterministic. Two fields are nullable by protocol: ``viewpoint`` is
``null`` until a viewpoint predictor exists and ``calibrated_confidence`` is ``null`` until a
calibration is fitted. ``cosine_similarity`` is a similarity in ``[-1, 1]``, never a
probability (§16 "never describe cosine similarity as probability"). ``versions`` is the
model/index identity the protocol requires in every response.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Decision = Literal["accepted", "uncertain", "no_match"]
DECISIONS: tuple[Decision, ...] = ("accepted", "uncertain", "no_match")
STAGES: tuple[str, ...] = ("detection", "segmentation", "embedding", "search")


class Viewpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    score: float = Field(ge=0.0, le=1.0)


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1)
    make: str | None
    model: str | None
    generation: str | None
    class_id: str
    cosine_similarity: float = Field(ge=-1.0, le=1.0)
    calibrated_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class StageLatency(BaseModel):
    """Wall-clock milliseconds per pipeline stage; every stage present, none negative."""

    model_config = ConfigDict(extra="forbid")

    detection: float = Field(ge=0.0)
    segmentation: float = Field(ge=0.0)
    embedding: float = Field(ge=0.0)
    search: float = Field(ge=0.0)


class Vehicle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detection_id: int = Field(ge=0)
    bbox_xyxy: list[float] = Field(min_length=4, max_length=4)
    detection_score: float = Field(ge=0.0, le=1.0)
    viewpoint: Viewpoint | None
    predictions: list[Prediction]
    decision: Decision
    latency_ms: StageLatency
    # optional extras, filled only when the request asks (``return_masks``/``return_embeddings``)
    mask_png_base64: str | None = None
    embedding: list[float] | None = None


class RecognizeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    profile_id: str
    index_version: str
    versions: dict[str, str | None]
    vehicles: list[Vehicle]


class ProfileInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: str
    track: str
    encoder: str
    variant: str
    dimension: str
    prototype_method: str
    collection: str
    index_version: str
    prototype_count: int
    default: bool


class ProfilesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_profile_id: str
    profiles: list[ProfileInfo]


class IndexesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    search_backend: str
    indexes: list[ProfileInfo]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "loading"]
    versions: dict[str, str | None]
