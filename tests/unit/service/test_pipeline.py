"""Scene image -> detections -> crops -> batched embedding -> search -> ranked results per car."""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from avr.preprocessing.crop import NEUTRAL_RGB
from avr.service.pipeline import RecognitionPipeline
from avr.service.schemas import STAGES, Vehicle
from avr.service.search import ExactSearch

from .fakes import ColourEncoder, ScriptedSegmenter, box_instance, scene

RED, GREEN, BLUE = (220, 20, 20), (20, 220, 20), (20, 20, 220)
SIZE = (120, 80)
BOX_A = (10, 10, 50, 40)
BOX_B = (60, 30, 110, 75)


def make_pipeline(instances, five_prototype_files, variant="p1", threshold=0.6):
    search = ExactSearch.from_parquet(five_prototype_files["parquet"])
    encoder = ColourEncoder()
    return (
        RecognitionPipeline(
            segmenter=ScriptedSegmenter(instances),
            encoder=encoder,
            search=search,
            variant=variant,
            accept_threshold=threshold,
        ),
        encoder,
    )


def test_no_detection_gives_empty_vehicles_and_complete_timings(five_prototype_files):
    pipeline, encoder = make_pipeline([], five_prototype_files)

    result = pipeline.recognize(scene(SIZE, {}), top_k=3)

    assert result.vehicles == []
    assert set(result.latency_ms) == set(STAGES)
    assert all(v >= 0.0 for v in result.latency_ms.values())
    assert encoder.calls == []  # nothing to embed


def test_one_vehicle_is_ranked_against_prototypes(five_prototype_files):
    inst = box_instance((10.4, 9.6, 50.2, 40.0), SIZE, score=0.93)
    pipeline, _ = make_pipeline([inst], five_prototype_files)

    result = pipeline.recognize(scene(SIZE, {BOX_A: RED}), top_k=3)

    assert len(result.vehicles) == 1
    v = result.vehicles[0]
    assert isinstance(v, Vehicle)
    assert v.detection_id == 0
    assert v.bbox_xyxy == [10.4, 9.6, 50.2, 40.0]  # input pixel coordinates, as detected
    assert v.detection_score == pytest.approx(0.93)
    assert v.viewpoint is None
    assert [p.rank for p in v.predictions] == [1, 2, 3]
    assert v.predictions[0].class_id == "c0"
    assert v.predictions[0].make == "A" and v.predictions[0].model == "a"
    assert v.predictions[0].generation == "g0"
    assert v.predictions[0].cosine_similarity == pytest.approx(1.0)
    assert v.predictions[0].calibrated_confidence is None
    assert v.decision == "accepted"
    assert v.latency_ms.model_dump().keys() == set(STAGES)


def test_multiple_vehicles_are_embedded_in_one_batch_in_detector_order(five_prototype_files):
    instances = [box_instance(BOX_B, SIZE, 0.8), box_instance(BOX_A, SIZE, 0.7)]
    pipeline, encoder = make_pipeline(instances, five_prototype_files)

    result = pipeline.recognize(scene(SIZE, {BOX_A: RED, BOX_B: BLUE}), top_k=2)

    assert [v.detection_id for v in result.vehicles] == [0, 1]
    assert [v.predictions[0].class_id for v in result.vehicles] == ["c2", "c0"]
    expected_boxes = [list(map(float, BOX_B)), list(map(float, BOX_A))]
    assert [v.bbox_xyxy for v in result.vehicles] == expected_boxes
    assert len(encoder.calls) == 1 and len(encoder.calls[0]) == 2
    assert all(len(v.predictions) == 2 for v in result.vehicles)


def test_p3_replaces_background_inside_the_crop_and_p1_keeps_it(five_prototype_files):
    # the instance mask covers only the inner half of the detected box
    inst = box_instance(BOX_A, SIZE)
    inst.mask[:] = False
    inst.mask[10:40, 10:30] = True
    image = scene(SIZE, {BOX_A: RED, (30, 10, 50, 40): GREEN})

    pipeline_p1, p1_encoder = make_pipeline([inst], five_prototype_files, variant="p1")
    pipeline_p1.recognize(image, top_k=1)
    pipeline_p3, p3_encoder = make_pipeline([inst], five_prototype_files, variant="p3")
    pipeline_p3.recognize(image, top_k=1)

    p1_crop = np.asarray(p1_encoder.calls[0][0])
    p3_crop = np.asarray(p3_encoder.calls[0][0])
    assert p1_crop.shape == p3_crop.shape  # same P1 geometry (expanded box), only masking differs
    # the crop box is BOX_A expanded by 5 % and rounded outwards: (8, 8, 52, 42)
    assert p1_crop.shape[:2] == (34, 44)
    # green half: kept by P1, neutral under P3
    assert tuple(p1_crop[15, 30]) == GREEN
    assert tuple(p3_crop[15, 30]) == NEUTRAL_RGB
    assert tuple(p3_crop[15, 10]) == RED


def test_top_k_is_clipped_to_the_prototype_count(five_prototype_files):
    pipeline, _ = make_pipeline([box_instance(BOX_A, SIZE)], five_prototype_files)

    result = pipeline.recognize(scene(SIZE, {BOX_A: RED}), top_k=50)

    assert len(result.vehicles[0].predictions) == 5


def test_decision_is_uncertain_below_the_cosine_threshold(five_prototype_files):
    pipeline, _ = make_pipeline([box_instance(BOX_A, SIZE)], five_prototype_files, threshold=1.5)

    result = pipeline.recognize(scene(SIZE, {BOX_A: RED}), top_k=1)

    assert result.vehicles[0].decision == "uncertain"


def test_optional_mask_and_embedding_are_returned_only_on_request(five_prototype_files):
    inst = box_instance(BOX_A, SIZE)
    pipeline, _ = make_pipeline([inst], five_prototype_files)
    image = scene(SIZE, {BOX_A: RED})

    plain = pipeline.recognize(image, top_k=1).vehicles[0]
    assert plain.mask_png_base64 is None and plain.embedding is None

    full = pipeline.recognize(image, top_k=1, return_masks=True, return_embeddings=True)
    v = full.vehicles[0]
    mask = np.asarray(Image.open(io.BytesIO(base64.b64decode(v.mask_png_base64)))) > 127
    assert mask.shape == (SIZE[1], SIZE[0])  # full input geometry
    assert np.array_equal(mask, inst.mask)
    assert len(v.embedding) == 5 and v.embedding[0] == pytest.approx(1.0)


def test_detections_below_the_minimum_score_are_dropped(five_prototype_files):
    instances = [box_instance(BOX_A, SIZE, 0.95), box_instance(BOX_B, SIZE, 0.2)]
    search = ExactSearch.from_parquet(five_prototype_files["parquet"])
    pipeline = RecognitionPipeline(
        segmenter=ScriptedSegmenter(instances),
        encoder=ColourEncoder(),
        search=search,
        variant="p1",
        min_detection_score=0.3,
    )

    result = pipeline.recognize(scene(SIZE, {BOX_A: RED, BOX_B: BLUE}), top_k=1)

    assert [v.detection_score for v in result.vehicles] == [pytest.approx(0.95)]
    assert result.vehicles[0].detection_id == 0
