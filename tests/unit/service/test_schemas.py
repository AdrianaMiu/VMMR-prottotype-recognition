"""Response schema of ``POST /v1/recognize`` (PROTOCOL.md §16 output skeleton)."""

import json

import pytest
from pydantic import ValidationError

from avr.service.schemas import (
    Prediction,
    RecognizeResponse,
    StageLatency,
    Vehicle,
)

SKELETON_KEYS = ["request_id", "profile_id", "index_version", "versions", "vehicles"]
VEHICLE_KEYS = [
    "detection_id",
    "bbox_xyxy",
    "detection_score",
    "viewpoint",
    "predictions",
    "decision",
    "latency_ms",
    "mask_png_base64",
    "embedding",
]
PREDICTION_KEYS = [
    "rank",
    "make",
    "model",
    "generation",
    "class_id",
    "cosine_similarity",
    "calibrated_confidence",
]
STAGES = ["detection", "segmentation", "embedding", "search"]


def _response(**overrides) -> RecognizeResponse:
    vehicle = Vehicle(
        detection_id=0,
        bbox_xyxy=[0.0, 0.0, 100.0, 100.0],
        detection_score=0.99,
        viewpoint=None,
        predictions=[
            Prediction(
                rank=1,
                make="Dacia",
                model="Logan",
                generation=None,
                class_id="dacia_logan_g2",
                cosine_similarity=0.81,
                calibrated_confidence=None,
            )
        ],
        decision="accepted",
        latency_ms=StageLatency(detection=1.0, segmentation=0.5, embedding=2.0, search=0.25),
    )
    fields = dict(
        request_id="4b8f1a0e-9e55-4d8d-a7a8-3b1b1f1f5a01",
        profile_id="track_a__dinov3_vitb16__p1__native__m0",
        index_version="a" * 64,
        versions={"encoder": "dinov3_vitb16"},
        vehicles=[vehicle],
    )
    fields.update(overrides)
    return RecognizeResponse(**fields)


def test_json_keys_follow_the_protocol_skeleton_in_order():
    payload = json.loads(_response().model_dump_json())

    assert list(payload) == SKELETON_KEYS
    assert list(payload["vehicles"][0]) == VEHICLE_KEYS
    assert list(payload["vehicles"][0]["predictions"][0]) == PREDICTION_KEYS
    assert list(payload["vehicles"][0]["latency_ms"]) == STAGES


def test_nullable_fields_serialise_as_null_not_omitted():
    payload = json.loads(_response().model_dump_json())
    vehicle = payload["vehicles"][0]

    assert vehicle["viewpoint"] is None
    assert vehicle["predictions"][0]["calibrated_confidence"] is None
    assert vehicle["predictions"][0]["generation"] is None


def test_empty_vehicles_is_a_valid_response():
    assert _response(vehicles=[]).vehicles == []


def test_serialisation_is_deterministic():
    assert _response().model_dump_json() == _response().model_dump_json()


@pytest.mark.parametrize(
    "bad",
    [
        {"detection": -1.0, "segmentation": 0.0, "embedding": 0.0, "search": 0.0},
        {"detection": 0.0, "segmentation": 0.0, "embedding": 0.0},
    ],
)
def test_latency_must_be_complete_and_non_negative(bad):
    with pytest.raises(ValidationError):
        StageLatency(**bad)


def test_cosine_similarity_is_bounded_and_rank_starts_at_one():
    with pytest.raises(ValidationError):
        Prediction(
            rank=0, make="a", model="b", generation=None, class_id="c", cosine_similarity=0.5
        )
    with pytest.raises(ValidationError):
        Prediction(
            rank=1, make="a", model="b", generation=None, class_id="c", cosine_similarity=1.5
        )


def test_decision_is_a_closed_vocabulary():
    vehicle = _response().vehicles[0].model_dump()
    with pytest.raises(ValidationError):
        Vehicle.model_validate({**vehicle, "decision": "probably"})
