"""The demo's code path on the real pipeline, once (GPU, checkpoint, dataset and data/ gated).

Runs ``run_demo`` in-process on the first Track A query example of the committed example list
with two profiles and checks the report is complete for every detected vehicle.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("rfdetr")
pytest.importorskip("gradio")

PROFILES = ["track_a__dinov3_vitb16__p1__native__m0", "track_a__dinov3_vitb16__p3__native__m0"]


@pytest.fixture(scope="module")
def recognizer():
    import torch

    from avr.config import load_config
    from avr.service.app import ServiceResources
    from avr.service.demo import InProcessRecognizer
    from avr.service.settings import ServiceSettings

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    cfg = load_config().segmentation
    cache = Path(os.environ.get("RF_HOME", "~/.roboflow/models")).expanduser()
    if not (cache / cfg.checkpoint.filename).exists():
        pytest.skip("RF-DETR checkpoint not downloaded")
    settings = ServiceSettings()
    if not (settings.prototypes_dir / "dinov3_vitb16" / "p1").is_dir():
        pytest.skip("M1 gate prototype tables not available under data/prototypes")
    if not settings.qdrant_dir.is_dir():
        pytest.skip("embedded Qdrant store not available under data/qdrant")
    resources = ServiceResources(settings)
    resources.load()
    yield InProcessRecognizer(resources)
    resources.close()


def test_real_demo_run_on_a_compcars_query_example(recognizer):
    from PIL import Image

    from avr.service.demo import EXAMPLES_JSON, compcars_examples, run_demo

    images = compcars_examples()
    if not images:
        pytest.skip("CompCars release or private path table not available")
    ids = json.loads(EXAMPLES_JSON.read_text())["image_ids"]
    assert len(images) <= len(ids)
    with Image.open(images[0]) as im:
        report = run_demo(recognizer, im.convert("RGB"), PROFILES, 5, True, True)
    assert len(report.recognitions) == 2
    for rec in report.recognitions:
        assert rec.vehicles
        assert all(len(v.predictions) == 5 for v in rec.vehicles)
    assert report.annotated.size == im.size
    assert "Vehicle 0" in report.vehicles_html
    assert len(report.latency) == 2
