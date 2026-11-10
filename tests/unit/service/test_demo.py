"""The Gradio demo: annotation, result formatting, session log and an in-process smoke test."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image

from avr.service.demo import (
    Detection,
    Recognition,
    SessionLog,
    annotate,
    colour_for,
    compcars_examples,
    decode_mask,
    detections_from,
    directory_examples,
    latency_rows,
    prediction_rows,
    run_demo,
    vehicles_html,
    versions_markdown,
)
from avr.service.schemas import Prediction, StageLatency, Vehicle

from .fakes import scene
from .test_app import PROFILE, SIZE, service  # noqa: F401  (service is a pytest fixture)

GREY = (100, 100, 100)


def test_annotate_blends_mask_and_draws_box_with_exact_pixels():
    image = Image.new("RGB", (60, 50), GREY)
    mask = np.zeros((50, 60), dtype=bool)
    mask[15:45, 10:50] = True
    out = annotate(image, [Detection(0, (10, 15, 50, 45), mask)])
    # detection 0 is the first palette colour (230, 25, 75); alpha 0.4:
    # 0.6*100+0.4*230 = 152, 0.6*100+0.4*25 = 70, 0.6*100+0.4*75 = 90
    assert colour_for(0) == (230, 25, 75)
    assert out.getpixel((40, 35)) == (152, 70, 90)  # inside the mask, off the outline
    assert out.getpixel((10, 40)) == (230, 25, 75)  # left outline pixel, full colour
    assert out.getpixel((5, 5)) == GREY  # background untouched
    assert image.getpixel((40, 35)) == GREY  # the input is not modified


def test_annotate_toggles_and_output_is_deterministic():
    image = Image.new("RGB", (60, 50), GREY)
    mask = np.zeros((50, 60), dtype=bool)
    mask[15:45, 10:50] = True
    det = [Detection(1, (10, 15, 50, 45), mask)]
    masks_only = annotate(image, det, show_masks=True, show_boxes=False)
    # colour 1 is (60, 180, 75): 0.6*100+0.4*60 = 84, 0.6*100+0.4*180 = 132, 0.6*100+0.4*75 = 90
    assert masks_only.getpixel((10, 40)) == (84, 132, 90)
    boxes_only = annotate(image, det, show_masks=False, show_boxes=True)
    assert boxes_only.getpixel((40, 35)) == GREY
    assert boxes_only.getpixel((10, 40)) == colour_for(1)
    neither = annotate(image, det, show_masks=False, show_boxes=False)
    assert list(neither.getdata()) == list(image.getdata())
    again = annotate(image, det)
    assert list(again.getdata()) == list(annotate(image, det).getdata())


def vehicle(det_id, class_ids, scores, decision="accepted", generation=None, mask_b64=None):
    return Vehicle(
        detection_id=det_id,
        bbox_xyxy=[10.0, 10.0, 50.0, 40.0],
        detection_score=0.93,
        viewpoint=None,
        predictions=[
            Prediction(
                rank=i + 1,
                make=f"Make{c}",
                model=f"Model{c}",
                generation=generation if i == 0 else None,
                class_id=c,
                cosine_similarity=s,
            )
            for i, (c, s) in enumerate(zip(class_ids, scores, strict=True))
        ],
        decision=decision,
        latency_ms=StageLatency(detection=11.0, segmentation=2.0, embedding=30.0, search=1.5),
        mask_png_base64=mask_b64,
    )


def recognitions():
    def rec(pid, a, b, lat):
        return Recognition(
            profile_id=pid,
            index_version="ab" * 32,
            versions={"encoder": "enc", "avr": "0.1", "segmenter": "seg"},
            vehicles=[a, b],
            latency_ms=lat,
        )

    lat1 = {"detection": 11.0, "segmentation": 2.0, "embedding": 30.0, "search": 1.5}
    lat2 = {"detection": 12.0, "segmentation": 2.5, "embedding": 31.0, "search": 2.0}
    return [
        rec(
            "track_a__p1",
            vehicle(0, ["001_0001", "001_0002"], [0.91, 0.5]),
            vehicle(1, ["002_0001"], [0.3], decision="uncertain"),
            lat1,
        ),
        rec(
            "track_b__p3",
            vehicle(0, ["g1", "g2"], [0.8, 0.45], generation="gen_x"),
            vehicle(1, ["g9"], [0.2], decision="no_match"),
            lat2,
        ),
    ]


def test_prediction_rows_format_rank_labels_similarity_and_decision():
    a, _ = recognitions()
    v0, v1 = a.vehicles
    assert prediction_rows(v0) == [
        ["1", "Make001_0001", "Model001_0001", "001_0001", "0.9100", "accepted"],
        ["2", "Make001_0002", "Model001_0002", "001_0002", "0.5000", ""],
    ]
    assert prediction_rows(v1)[0][-1] == "uncertain"
    b, _ = recognitions()[1].vehicles
    assert prediction_rows(b)[0][3] == "gen_x"  # provisional generation beats the class id


def test_latency_rows_have_one_row_per_profile_with_a_total():
    rows = latency_rows(recognitions())
    assert rows[0] == ["track_a__p1", 11.0, 2.0, 30.0, 1.5, 44.5]
    assert rows[1] == ["track_b__p3", 12.0, 2.5, 31.0, 2.0, 47.5]


def test_vehicles_html_has_each_vehicle_with_both_profiles_side_by_side():
    page = vehicles_html(recognitions())
    assert page.count('class="avr-vehicle"') == 2
    assert page.count('class="avr-card"') == 4
    assert page.index("Vehicle 0") < page.index("Vehicle 1")
    assert "track_a__p1" in page and "track_b__p3" in page and "0.9100" in page


def test_vehicles_html_escapes_labels_and_handles_no_vehicles():
    rec = recognitions()[0]
    rec.vehicles[0].predictions[0] = (
        rec.vehicles[0].predictions[0].model_copy(update={"make": "<script>x</script>"})
    )
    assert "<script>" not in vehicles_html([rec])
    empty = Recognition("p", "0" * 64, {}, [], {})
    assert "No vehicle detected" in vehicles_html([empty])


def test_versions_markdown_lists_profiles_index_and_disclaimer():
    text = versions_markdown(recognitions())
    assert "track_a__p1" in text and ("ab" * 8) in text
    assert "cosine similarity is not a probability" in text.lower()


def test_mask_round_trip_through_the_service_encoding():
    from avr.service.pipeline import mask_to_png_base64

    mask = np.zeros((6, 8), dtype=bool)
    mask[2:4, 1:5] = True
    (det,) = detections_from([vehicle(0, ["c"], [0.5], mask_b64=mask_to_png_base64(mask))])
    assert np.array_equal(det.mask, mask)
    assert np.array_equal(decode_mask(mask_to_png_base64(mask)), mask)


class FakeRecognizer:
    def __init__(self):
        self.calls = []

    def recognize(self, image, profile_id, top_k, return_masks):
        self.calls.append((profile_id, top_k, return_masks))
        return next(r for r in recognitions() if r.profile_id == profile_id)


def test_run_demo_compares_profiles_and_writes_the_opt_in_log(tmp_path):
    fake = FakeRecognizer()
    log_path = tmp_path / "logs" / "session.jsonl"
    image = Image.new("RGB", (80, 60), GREY)
    report = run_demo(
        fake, image, ["track_a__p1", "track_b__p3"], 5, True, True, SessionLog(log_path)
    )
    assert fake.calls == [("track_a__p1", 5, True), ("track_b__p3", 5, False)]
    assert report.annotated.size == (80, 60)
    assert len(report.latency) == 2
    lines = [json.loads(x) for x in log_path.read_text().splitlines()]
    assert [x["profile"] for x in lines] == ["track_a__p1", "track_b__p3"]
    assert lines[0]["vehicles"] == 2 and lines[0]["top1"][0]["class_id"] == "001_0001"
    assert lines[0]["timestamp"]


def test_session_log_is_off_without_a_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    SessionLog(None).write(recognitions()[0])
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError):
        run_demo(FakeRecognizer(), Image.new("RGB", (4, 4)), [], 5, True, True)


def test_examples_resolve_image_ids_through_the_path_table(tmp_path):
    import pandas as pd

    root = tmp_path / "compcar"
    (root / "image" / "1" / "2" / "2010").mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(root / "image/1/2/2010/a.jpg")
    table = tmp_path / "paths.parquet"
    pd.DataFrame(
        {
            "image_id": ["id_a", "id_b"],
            "relative_path": ["image/1/2/2010/a.jpg", "image/9/9/9/b.jpg"],
        }
    ).to_parquet(table)
    listing = tmp_path / "examples.json"
    listing.write_text(json.dumps({"image_ids": ["id_a", "id_b", "id_missing"]}))
    found = compcars_examples(listing, table, root)
    assert found == [root / "image/1/2/2010/a.jpg"]  # missing file and unknown id are skipped
    assert compcars_examples(tmp_path / "none.json", table, root) == []

    photos = tmp_path / "photos"
    photos.mkdir()
    (photos / "b.png").write_bytes(b"x")
    (photos / "a.jpg").write_bytes(b"x")
    (photos / "notes.txt").write_text("x")
    assert [p.name for p in directory_examples(photos)] == ["a.jpg", "b.png"]
    assert directory_examples(None) == []


def test_in_process_recognizer_and_gradio_smoke(service, tmp_path):  # noqa: F811
    pytest.importorskip("gradio")
    from gradio_client import Client, handle_file

    from avr.service.demo import InProcessRecognizer, build_demo

    from .fakes import box_instance

    box_a, box_b = (10, 10, 50, 40), (60, 30, 110, 75)
    _, resources = service([box_instance(box_a, SIZE), box_instance(box_b, SIZE)])
    resources.load()
    recognizer = InProcessRecognizer(resources)
    listing = recognizer.profiles()
    assert listing.default_profile_id == PROFILE

    image = scene(SIZE, {box_a: (220, 20, 20), box_b: (20, 20, 220)})
    report = run_demo(recognizer, image, [PROFILE], 3, True, True)
    assert [v.detection_id for v in report.recognitions[0].vehicles] == [0, 1]
    assert report.recognitions[0].vehicles[0].predictions[0].class_id == "c0"  # red -> axis 0

    path = tmp_path / "scene.png"
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    path.write_bytes(buffer.getvalue())
    log_path = tmp_path / "s.jsonl"
    demo = build_demo(recognizer, [path], SessionLog(log_path))
    _, url, _ = demo.launch(prevent_thread_lock=True, server_port=None, quiet=True)
    try:
        client = Client(url, verbose=False)
        annotated, table_html, latency, versions = client.predict(
            handle_file(str(path)), [PROFILE], 3, True, True, api_name="/recognize"
        )
        assert "Vehicle 0" in table_html and "Vehicle 1" in table_html
        assert "not a probability" in versions
        assert annotated
    finally:
        demo.close()
    assert json.loads(log_path.read_text().splitlines()[0])["vehicles"] == 2
